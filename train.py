import argparse
import os
import torch.optim as optim
import torch.nn as nn
import torch

from model.fewshot import ProtoFormer
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from common.amp import get_device, make_scaler, resolve_amp_dtype, amp_context


def train(epoch, model, dataloader, optimizer, training, amp_dtype=None, scaler=None):
    r""" Train ProtoFormer """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.train() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        if training:
            with amp_context(amp_dtype):
                logit_mask = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
                loss = model.compute_loss(logit_mask, batch['query_mask'])
        else:
            with torch.no_grad(), amp_context(amp_dtype):
                logit_mask = model.predict_mask(batch)
        pred_mask=(logit_mask>0.5).float().squeeze(1)
        if training:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        if training:
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=100)
        else:
            average_meter.update(area_inter, area_union, batch['class_id'],loss=None)
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=100)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='ProtoFormer Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../data')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='log_protoformer_r50_coco0')
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=30)
    parser.add_argument('--nworker', type=int, default=4)
    parser.add_argument('--reduce_dim', type=int, default=64)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--shot',type=int, default=1)
    parser.add_argument('--amp', type=str, default='auto', choices=['auto', 'bf16', 'fp16', 'off'], help='AMP dtype; auto: bf16 on Ampere+ (L4), fp16 below, off on CPU')
    parser.add_argument('--sam2_checkpoint', type=str, default=None, help='SAM2 hiera checkpoint .pth (optional)')
    parser.add_argument('--patience', type=int, default=20, help='early stop if val mIoU not improve for N val checks; 0 = off')
    parser.add_argument('--val_interval', type=int, default=3, help='run validation every N epochs; best model & early stop only evaluated on val epochs')
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = ProtoFormer(shot=args.shot,reduce_dim=args.reduce_dim,checkpoint_path=args.sam2_checkpoint)
    Logger.log_params(model)
    # Device setup
    device = get_device()
    amp_dtype = resolve_amp_dtype(device, args.amp)
    scaler = make_scaler(amp_dtype)
    Logger.info(f'# device: {device}  amp: {amp_dtype}  scaler: {scaler is not None}')
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=473, datapath=args.datapath)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn',shot=args.shot)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, 1, args.nworker, args.fold, 'val',shot=args.shot)

    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    val_checks_no_improve = 0

    for epoch in range(args.niter):
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True, amp_dtype=amp_dtype, scaler=scaler)

        val_loss = val_miou = val_fb_iou = None
        if args.val_interval > 0 and ((epoch + 1) % args.val_interval == 0 or epoch == args.niter - 1):
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False, amp_dtype=amp_dtype)

            # Save the best model — only on val epochs
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(model, epoch, val_miou)
                val_checks_no_improve = 0
            else:
                val_checks_no_improve += 1
                if args.patience > 0 and val_checks_no_improve >= args.patience:
                    Logger.info(f'Early stop: val mIoU no improve for {val_checks_no_improve} val checks @ epoch {epoch}')
                    break

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')

r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def write_result(self, split, epoch):
        iou, fb_iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath + '.log')
        cls.benchmark = args.benchmark
        if not os.path.exists(cls.logpath): 
            os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Semantic Segmentation with ProtoFormer ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        # Payload berbentuk dict supaya epoch/mIoU ikut tersimpan. Format lama (raw
        # state_dict, sebelum commit ini) tetap bisa dimuat lewat load_weights() di bawah.
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'val_miou': float(val_miou),
        }, os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @staticmethod
    def _unwrap(payload):
        if isinstance(payload, torch.nn.Module):
            return payload.state_dict(), {'source': 'whole_module'}
        if isinstance(payload, dict) and 'model' in payload and isinstance(payload['model'], (dict, torch.nn.Module)):
            inner = payload['model']
            state = inner.state_dict() if isinstance(inner, torch.nn.Module) else inner
            return state, {k: v for k, v in payload.items() if k != 'model'}
        # raw state_dict lama: semua value adalah tensor
        return payload, {}

    @classmethod
    def load_weights(cls, model, path, device='cpu', strict=True):
        r""" Muat weight dari checkpoint lama maupun baru. Return info dict. """
        payload = torch.load(path, map_location=device, weights_only=False)
        state, info = cls._unwrap(payload)
        try:
            model.load_state_dict(state, strict=strict)
        except RuntimeError as e:
            # pesan default sulit dibaca; tampilkan ringkas supaya jelas apa yang beda
            own = set(model.state_dict().keys())
            got = set(state.keys())
            missing, unexpected = sorted(own - got), sorted(got - own)
            raise RuntimeError(
                'checkpoint tidak kompatibel dengan model.\n'
                '  file      : %s\n  missing   : %d %s\n  unexpected: %d %s\n  detail: %s'
                % (path, len(missing), missing[:5], len(unexpected), unexpected[:5], e)
            ) from e
        return info

    @classmethod
    def save_resume(cls, model, optimizer, scaler, epoch, best_val_miou, val_checks_no_improve, args):
        r""" Checkpoint resume: semua state yang dibutuhkan untuk lanjut di epoch berikutnya """
        payload = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'epoch': epoch,                       # epoch yang baru selesai
            'best_val_miou': best_val_miou,
            'val_checks_no_improve': val_checks_no_improve,
            'args': vars(args),                   # untuk validasi saat resume
        }
        # tulis ke .tmp lalu rename: proses mati di tengah torch.save tidak merusak checkpoint lama
        path = os.path.join(cls.logpath, 'last_model.pt')
        tmp = path + '.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load_resume(cls, model, optimizer, scaler, device):
        r""" Muat last_model.pt kalau ada. Return dict state atau None.

        Hanya menerima checkpoint yang punya 'epoch' (format resume). File best_model.pt
        lama (raw state_dict) sengaja ditolak: tanpa optimizer/epoch, resume tidak bermakna
        dan train.py akan KeyError. Pakai load_weights() untuk file seperti itu.
        """
        path = os.path.join(cls.logpath, 'last_model.pt')
        if not os.path.isfile(path):
            return None
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state, info = cls._unwrap(ckpt)
        if 'epoch' not in info or 'args' not in info:
            cls.info('# resume: %s bukan checkpoint resume (tidak ada epoch/args), dilewati' % path)
            return None
        model.load_state_dict(state)
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        # default supaya checkpoint lama-tanpa-field-baru tidak KeyError di train.py
        info.setdefault('best_val_miou', float('-inf'))
        info.setdefault('val_checks_no_improve', 0)
        return {'model': state, **info}

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']: 
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))


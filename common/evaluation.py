r""" Evaluate mask prediction """
import torch


class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_mask, batch):
        gt_mask = batch.get('query_mask')

        # pred_mask must match gt_mask shape (letterbox crop to original size)
        if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
            import torch.nn.functional as F
            pred_mask = F.interpolate(pred_mask.unsqueeze(1).float(),
                                      size=gt_mask.shape[-2:],
                                      mode='nearest').squeeze(1)

        # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None:
            if query_ignore_idx.shape[-2:] != gt_mask.shape[-2:]:
                import torch.nn.functional as F
                query_ignore_idx = F.interpolate(query_ignore_idx.unsqueeze(1).float(),
                                                 size=gt_mask.shape[-2:],
                                                 mode='nearest').squeeze(1).to(query_ignore_idx.dtype)
            assert torch.logical_and(query_ignore_idx.long(), gt_mask.long()).sum() == 0
            query_ignore_idx = query_ignore_idx.long() * cls.ignore_index
            gt_mask = gt_mask.clone().long() + query_ignore_idx
            pred_mask[gt_mask == cls.ignore_index] = cls.ignore_index

        # compute intersection and union of each episode in a batch
        # torch.histc unreliable on bf16/half; evaluate in fp32 on same device
        pred_mask = pred_mask.float()
        gt_mask = gt_mask.float()
        area_inter, area_pred, area_gt = [],  [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            _inter = _pred_mask[_pred_mask == _gt_mask]
            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union

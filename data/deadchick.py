r""" Dead-chick few-shot segmentation dataset (single class: dead=1).

Layout expected in datapath (one of):
  - datapath/dead-chick/*.jpg + *_mask.png            (repo datasets/ layout)
  - datapath/deadchick/*.jpg + *_mask.png
  - datapath itself contains *.jpg + *_mask.png

Mask values: 0=background, 1=dead (binary, see _classes.csv).
Episode sampling follows pascal.py:
  - query = image at idx (every image is queried once in eval)
  - support = `shot` random OTHER images (never the query itself)
  - class_id = 1 (dead)
"""
import glob
import os

import PIL.Image as Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DatasetDeadChick(Dataset):
    """Dead-chick loader with letterbox (black padding) preprocessing.

    Padding mode (default): resize preserving aspect ratio, pad remaining
    space with black (RGB) / 0 (mask). No stretching.

    img_size is taken from the transform pipeline: FSSDataset.initialize()
    builds Resize(img_size). We recover it from the Resize transform and
    need to feed Pillow images ourselves so padding is applied BEFORE
    transform. The transform then sees an already-square Pillow image.
    """

    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 1
        self.nclass = 1  # only 'dead'
        self.benchmark = 'deadchick'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.transform = transform
        # recover img_size from transform (Resize in Compose)
        self.img_size = getattr(transform, 'transforms', [None])[0].size[0] if hasattr(transform, 'transforms') else 512
        if isinstance(self.img_size, tuple):
            self.img_size = self.img_size[0]

        self.base_path = self._find_base(datapath)
        self.class_ids = [1]  # dead
        self.img_metadata = self.build_img_metadata()  # [[name, class_id], ...]
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    # ---------- helpers ----------
    @staticmethod
    def _find_base(datapath):
        for cand in (os.path.join(datapath, 'dead-chick'),
                     os.path.join(datapath, 'deadchick'),
                     datapath):
            if glob.glob(os.path.join(cand, '*.jpg')):
                return cand
        raise FileNotFoundError(
            f'dead-chick jpg not found under {datapath!r} (looked in dead-chick/, deadchick/, datapath/)')

    def __len__(self):
        # Eval: one episode per image. Train: repeat len to allow epochs; ratio handled by DataLoader.
        if self.split == 'trn':
            return len(self.img_metadata) * 10
        return len(self.img_metadata)

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img_pil, query_cmask, support_pils, support_cmasks, org_qry_imsize = \
            self.load_frame(query_name, support_names)

        # letterbox -> transform (Resize already square, so no stretch here)
        query_img = self.transform(query_img_pil)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(),
                                        query_img.size()[-2:], mode='nearest').squeeze()
        # mask already letterboxed to size x size; interpolate is identity when sizes match,
        # but keep it for safety (e.g. img_size 512 vs transform Resize 512).
        # binary mask (0/1) -> protoformer expects 0/1 for class_id=1 path (see pascal.extract_ignore_idx)
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs = torch.stack([self.transform(s) for s in support_pils])
        support_masks, support_ignore_idxs = [], []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(),
                                   support_imgs.size()[-2:], mode='nearest').squeeze()
            sm, si = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(sm)
            support_ignore_idxs.append(si)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        return {'query_img': query_img,
                'query_mask': query_mask,
                'query_name': query_name,
                'query_ignore_idx': query_ignore_idx,
                'org_query_imsize': org_qry_imsize,
                'support_imgs': support_imgs,
                'support_masks': support_masks,
                'support_names': support_names,
                'support_ignore_idxs': support_ignore_idxs,
                'class_id': torch.tensor(class_sample)}

    @staticmethod
    def extract_ignore_idx(mask, class_id):
        """mask already 0/1 -> binary mask + ignore-index compatible with pascal.py.

        pascal.py: boundary = (mask/255).floor(); mask[mask!=class_id+1]=0; mask[mask==class_id+1]=1
        Here class_id=1, mask values 0/1, so 'dead' pixels are 1 directly.
        """
        boundary = torch.zeros_like(mask)
        out = (mask == 1).float()
        return out, boundary

    # ---------- letterbox helpers (Pillow, before transform) ----------
    @staticmethod
    def _letterbox_img(pil, size):
        """Resize preserving aspect ratio, pad with black to size x size."""
        w, h = pil.size
        scale = size / max(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = pil.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (size, size), (0, 0, 0))
        canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
        return canvas

    @staticmethod
    def _letterbox_mask(np_mask, size):
        """Resize preserving aspect ratio, pad with 0 to size x size (numpy HxW)."""
        h, w = np_mask.shape[:2]
        scale = size / max(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        from PIL import Image as _I
        resized = _I.fromarray(np_mask).resize((nw, nh), _I.NEAREST)
        canvas = np.zeros((size, size), dtype=np_mask.dtype)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = np.array(resized)
        return canvas

    @staticmethod
    def _unletterbox(pred_logit, orig_w, orig_h, size):
        """Crop centre region of square pred back to original aspect, then resize to orig."""
        # pred_logit: (H,W) tensor size x size (float)
        scale = size / max(orig_w, orig_h)
        nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        crop = pred_logit[:, y0:y0 + nh, x0:x0 + nw] if pred_logit.dim() == 3 else pred_logit[y0:y0 + nh, x0:x0 + nw]
        if crop.dim() == 2:
            crop = crop.unsqueeze(0).unsqueeze(0)
        elif crop.dim() == 3:
            crop = crop.unsqueeze(0)
        out = F.interpolate(crop.float(), size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return out.squeeze()

    def load_frame(self, query_name, support_names):
        q_img_raw = self.read_img(query_name)          # Pil
        q_mask_np = self.read_mask_np(query_name)       # np HxW
        s_raw = [self.read_img(n) for n in support_names]
        s_mask_np = [self.read_mask_np(n) for n in support_names]
        org = q_img_raw.size                             # (w,h) original
        # letterboxed versions for the model input
        q_img = self._letterbox_img(q_img_raw, self.img_size)
        q_cm = torch.tensor(self._letterbox_mask(q_mask_np, self.img_size))
        s_imgs = [self._letterbox_img(im, self.img_size) for im in s_raw]
        s_cms = [torch.tensor(self._letterbox_mask(m, self.img_size)) for m in s_mask_np]
        return q_img, q_cm, s_imgs, s_cms, org

    def read_mask_np(self, img_name):
        path = os.path.join(self.base_path, img_name + '_mask.png')
        return np.array(Image.open(path))

    def read_mask(self, img_name):
        path = os.path.join(self.base_path, img_name + '_mask.png')
        return torch.tensor(np.array(Image.open(path)))

    def read_img(self, img_name):
        return Image.open(os.path.join(self.base_path, img_name + '.jpg')).convert('RGB')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]
        support_names = []
        while len(support_names) < self.shot:
            s = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if s != query_name:
                support_names.append(s)
        return query_name, support_names, class_sample

    def build_img_metadata(self):
        names = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(self.base_path, '*.jpg')))
        # sanity: every jpg must have a mask
        meta = []
        for name in names:
            mp = os.path.join(self.base_path, name + '_mask.png')
            if os.path.exists(mp):
                meta.append([name, 1])
        print(f'Total ({self.split}) dead-chick images are : {len(meta)}')
        return meta

    def build_img_metadata_classwise(self):
        d = {1: []}
        for name, cls in self.img_metadata:
            d[cls].append(name)
        return d

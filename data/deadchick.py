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
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 1
        self.nclass = 1  # only 'dead'
        self.benchmark = 'deadchick'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.transform = transform

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
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = \
            self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(),
                                        query_img.size()[-2:], mode='nearest').squeeze()
        # binary mask (0/1) -> protoformer expects 0/1 for class_id=1 path (see pascal.extract_ignore_idx)
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs = torch.stack([self.transform(s) for s in support_imgs])
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

    def load_frame(self, query_name, support_names):
        q_img = self.read_img(query_name)
        q_mask = self.read_mask(query_name)
        s_imgs = [self.read_img(n) for n in support_names]
        s_masks = [self.read_mask(n) for n in support_names]
        return q_img, q_mask, s_imgs, s_masks, q_img.size

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

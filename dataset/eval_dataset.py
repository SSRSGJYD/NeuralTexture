import numpy as np
import os
import torch
from torch.utils.data import Dataset

from util import map_transform


class EvalDataset(Dataset):

    def __init__(self, dir, idx_list):
        self.idx_list = idx_list
        self.dir = dir

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        uv_map = np.load(os.path.join(self.dir, 'uv_'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')

        # final transform
        uv_map = map_transform(uv_map)
        # mask for invalid uv positions
        mask = torch.max(uv_map, dim=2)[0].le(-1.0 + 1e-6)
        mask = mask.repeat((3, 1, 1))

        return uv_map, mask, self.idx_list[idx]

    def collect_fn(data):
        uv_maps, masks, idxs = zip(*data)
        uv_maps = torch.stack(tuple(uv_maps), dim=0)
        masks = torch.stack(tuple(masks), dim=0)
        return uv_maps, masks, idxs

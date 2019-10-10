import numpy as np
import os
import torch
from torch.utils.data import Dataset

from util import map_transform, view2sh


class EvalDataset(Dataset):

    def __init__(self, dir, idx_list, view_direction=False):
        self.idx_list = idx_list
        self.dir = dir
        self.view_direction = view_direction

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

        if self.view_direction:
            view_map = np.load(os.path.join(self.dir, 'view_' + self.idx_list[idx] + '.npy'))
            sh_map = view2sh(view_map, 0, view_map.shape[0], 0, view_map.shape[1])
            sh_map = map_transform(sh_map)
            return uv_map, sh_map, mask, self.idx_list[idx]
        else:
            return uv_map, mask, self.idx_list[idx]

    def collect_fn(data, view_direction=False):
        if view_direction:
            uv_maps, sh_maps, masks, idxs = zip(*data)
            uv_maps = torch.stack(tuple(uv_maps), dim=0)
            sh_maps = torch.stack(tuple(sh_maps), dim=0)
            masks = torch.stack(tuple(masks), dim=0)
            return uv_maps, sh_maps, masks, idxs
        else:
            uv_maps, masks, idxs = zip(*data)
            uv_maps = torch.stack(tuple(uv_maps), dim=0)
            masks = torch.stack(tuple(masks), dim=0)
            return uv_maps, masks, idxs

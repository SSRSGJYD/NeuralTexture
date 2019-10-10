import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

from util import augment, augment_view


class UVDataset(Dataset):

    def __init__(self, dir, idx_list, H, W, view_direction=False):
        self.idx_list = idx_list
        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.idx_list[idx]+'.ppm'), 'r')
        uv_map = np.load(os.path.join(self.dir, 'uv_'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        if self.view_direction:
            view_map = np.load(os.path.join(self.dir, 'view_'+self.idx_list[idx]+'.npy'))
            img, uv_map, mask, sh_map = augment_view(img, uv_map, view_map, self.crop_size)
            return img, uv_map, view_map, mask
        else:
            img, uv_map, mask = augment(img, uv_map, self.crop_size)
            return img, uv_map, mask

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

from util import augment


class UVDataset(Dataset):

    def __init__(self, dir, idx_list, H, W, view_direction=False):
        self.idx_list = idx_list
        self.dir = dir
        self.crop_size = (H, W)
        self.view_direction = view_direction

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, 'frame/'+self.idx_list[idx]+'.png'), 'r')
        uv_map = np.load(os.path.join(self.dir, 'uv/'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        img, uv_map, mask = augment(img, uv_map, self.crop_size)
        if self.view_direction:
            # view_map = np.load(os.path.join(self.dir, 'view_normal/'+self.idx_list[idx]+'.npy'))
            extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
            return img, uv_map, extrinsics, mask
        else:
            
            return img, uv_map, mask

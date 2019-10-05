import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import os
from PIL import Image

from util import augment


class UVDataset(Dataset):

    def __init__(self, dir, idx_list, H, W):
        self.idx_list = idx_list
        self.dir = dir
        self.crop_size = (H, W)

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
        img, uv_map, mask = augment(img, uv_map, self.crop_size)
        return img, uv_map, mask

    def collect_fn(data):
        images, uv_maps, masks = zip(*data)
        images = torch.cat(images, dim=0)
        uv_maps = torch.cat(uv_maps, dim=0)
        masks = torch.cat(masks, dim=0)
        return images, uv_maps, masks

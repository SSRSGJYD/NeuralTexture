import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import os
from PIL import Image

from util import augment


class UVDataset(Dataset):

    def __init__(self, dir, idx_list, W, H):
        self.idx_list = idx_list
        self.dir = dir
        self.crop_size = (W, H)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir), self.idx_list[idx]+'.ppm')
        uv_map = np.load(os.path.join(self.dir), 'uv_'+self.idx_list[idx]+'.npy')
        img = augment(img, uv_map, self.crop_size)
        return img, uv_map

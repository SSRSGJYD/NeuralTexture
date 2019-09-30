import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')

from model.texture import Texture
from model.unet import UNet


class Renderer(nn.Module):
    def __init__(self, W, H, input_channels, output_channels):
        super(Renderer, self).__init__()
        self.texture = Texture(W, H)
        self.unet = UNet(input_channels, output_channels)

    def forward(self, x):
        x = self.texture(x)
        x = self.unet(x)
        return x


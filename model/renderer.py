import sys
import torch
import torch.nn as nn

sys.path.append('..')
from model.texture import Texture
from model.unet import UNet


class Renderer(nn.Module):
    def __init__(self, W, H, pyramid_num, view_direction=False):
        super(Renderer, self).__init__()
        self.pyramid_num = pyramid_num
        self.view_direction = view_direction
        self.texture = Texture(W, H, pyramid_num)
        if self.view_direction:
            self.unet = UNet(pyramid_num, 3)
        else:
            self.unet = UNet(pyramid_num+9, 3)

    def forward(self, uv_map, sh_map=None):
        x = self.texture(uv_map)
        if self.view_direction:
            x = torch.cat((x, sh_map), dim=1)
        y = self.unet(x)
        return y


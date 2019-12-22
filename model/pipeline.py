import sys
import torch
import torch.nn as nn

sys.path.append('..')
from model.texture import Texture
from model.unet import UNet


class PipeLine(nn.Module):
    def __init__(self, W, H, feature_num, use_pyramid=True, view_direction=True, residual=False):
        super(PipeLine, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.view_direction = view_direction
        self.residual = residual
        self.texture = Texture(W, H, feature_num, use_pyramid)
        self.unet = UNet(feature_num, 3)

    def forward(self, *args):
        if self.view_direction:
            uv_map, sh_map = args
            x = self.texture(uv_map)
            assert x.shape[1] >= 12
            x[:, 3:12, :, :] = x[:, 3:12, :, :] * sh_map[:, :, :, :]
        else:
            uv_map = args[0]
            x = self.texture(uv_map)
        y = self.unet(x)
        if self.residual:
            y = x[:, 0:3, :, :] + y

        return x[:, 0:3, :, :], y
import torch
import torch.nn as nn
import torch.nn.functional as F


class Texture(nn.Module):
    def __init__(self, W, H):
        super(Texture, self).__init__()
        self.layer1 = torch.FloatTensor(W, H)
        self.layer2 = torch.FloatTensor(W // 2, H // 2)
        self.layer3 = torch.FloatTensor(W // 4, H // 4)
        self.layer4 = torch.FloatTensor(W // 8, H // 8)

    def forward(self, x):
        x = x * 0.5 + 0.5
        y1 = F.grid_sample(self.layer1, x)
        y2 = F.grid_sample(self.layer2, x)
        y3 = F.grid_sample(self.layer3, x)
        y4 = F.grid_sample(self.layer4, x)
        y = y1 + y2 + y3 + y4
        return y

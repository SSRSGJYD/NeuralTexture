import torch
import torch.nn as nn
import torch.nn.functional as F


class Texture(nn.Module):
    def __init__(self, W, H):
        super(Texture, self).__init__()
        self.layer1 = torch.FloatTensor(1, 1, W, H)
        self.layer1.requires_grad = True
        # self.layer1.group_name = 'layer1'
        self.layer2 = torch.FloatTensor(1, 1, W // 2, H // 2)
        self.layer2.requires_grad = True
        # self.layer2.group_name = 'layer2'
        self.layer3 = torch.FloatTensor(1, 1, W // 4, H // 4)
        self.layer3.requires_grad = True
        # self.layer3.group_name = 'layer3'
        self.layer4 = torch.FloatTensor(1, 1, W // 8, H // 8)
        self.layer4.requires_grad = True
        # self.layer4.group_name = 'layer4'

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y1 = F.grid_sample(self.layer1.repeat(batch,1,1,1), x)
        y2 = F.grid_sample(self.layer2.repeat(batch,1,1,1), x)
        y3 = F.grid_sample(self.layer3.repeat(batch,1,1,1), x)
        y4 = F.grid_sample(self.layer4.repeat(batch,1,1,1), x)
        y = y1 + y2 + y3 + y4
        return y

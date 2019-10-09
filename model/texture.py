import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianPyramid(nn.Module):
    def __init__(self, W, H):
        super(LaplacianPyramid, self).__init__()
        self.layer1 = nn.Parameter(torch.FloatTensor(1, 1, W, H))
        self.layer2 = nn.Parameter(torch.FloatTensor(1, 1, W // 2, H // 2))
        self.layer3 = nn.Parameter(torch.FloatTensor(1, 1, W // 4, H // 4))
        self.layer4 = nn.Parameter(torch.FloatTensor(1, 1, W // 8, H // 8))

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y1 = F.grid_sample(self.layer1.repeat(batch,1,1,1), x)
        y2 = F.grid_sample(self.layer2.repeat(batch,1,1,1), x)
        y3 = F.grid_sample(self.layer3.repeat(batch,1,1,1), x)
        y4 = F.grid_sample(self.layer4.repeat(batch,1,1,1), x)
        y = y1 + y2 + y3 + y4
        return y


class Texture(nn.Module):
    def __init__(self, W, H):
        super(Texture, self).__init__()
        self.pyramid1 = LaplacianPyramid(W, H)
        self.pyramid2 = LaplacianPyramid(W, H)
        self.pyramid3 = LaplacianPyramid(W, H)

    def forward(self, x):
        y1 = self.pyramid1(x)
        y2 = self.pyramid1(x)
        y3 = self.pyramid1(x)
        y = torch.cat((y1, y2, y3), dim=1)
        return y

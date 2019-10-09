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
    def __init__(self, W, H, pyramid_num):
        super(Texture, self).__init__()
        self.pyramid_num = pyramid_num
        self.pyramids = nn.ModuleList([LaplacianPyramid(W, H) for i in range(pyramid_num)])
        self.layer1 = nn.ParameterList()
        self.layer2 = nn.ParameterList()
        self.layer3 = nn.ParameterList()
        self.layer4 = nn.ParameterList()
        for i in range(self.pyramid_num):
            self.layer1.append(self.pyramids[i].layer1)
            self.layer2.append(self.pyramids[i].layer2)
            self.layer3.append(self.pyramids[i].layer3)
            self.layer4.append(self.pyramids[i].layer4)

    def forward(self, x):
        y_i = []
        for i in range(self.pyramid_num):
            y = self.pyramids[i](x)
            y_i.append(y)
        y = torch.cat(tuple(y_i), dim=1)
        return y

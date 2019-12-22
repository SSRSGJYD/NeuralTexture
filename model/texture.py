import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleLayerTexture(nn.Module):
    def __init__(self, W, H):
        super(SingleLayerTexture, self).__init__()
        self.layer1 = nn.Parameter(torch.FloatTensor(1, 1, W, H))

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y = F.grid_sample(self.layer1.repeat(batch,1,1,1), x)
        return y


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
    def __init__(self, W, H, feature_num, use_pyramid=True):
        super(Texture, self).__init__()
        self.feature_num = feature_num
        self.use_pyramid = use_pyramid
        self.layer1 = nn.ParameterList()
        self.layer2 = nn.ParameterList()
        self.layer3 = nn.ParameterList()
        self.layer4 = nn.ParameterList()
        if self.use_pyramid:
            self.textures = nn.ModuleList([LaplacianPyramid(W, H) for i in range(feature_num)])
            for i in range(self.feature_num):
                self.layer1.append(self.textures[i].layer1)
                self.layer2.append(self.textures[i].layer2)
                self.layer3.append(self.textures[i].layer3)
                self.layer4.append(self.textures[i].layer4)
        else:
            self.textures = nn.ModuleList([SingleLayerTexture(W, H) for i in range(feature_num)])
            for i in range(self.feature_num):
                self.layer1.append(self.textures[i].layer1)
        
    def forward(self, x):
        y_i = []
        for i in range(self.feature_num):
            y = self.textures[i](x)
            y_i.append(y)
        y = torch.cat(tuple(y_i), dim=1)
        return y

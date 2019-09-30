import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from dataset.uv_dataset import UVDataset
from model.renderer import Renderer

parser = argparse.ArgumentParser()

parser.add_argument('--pyramidw', type=int, default=config.PYRAMID_W)
parser.add_argument('--pyramidh', type=int, default=config.PYRAMID_H)

parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
parser.add_argument('--train', default=config.TRAIN_SET)
parser.add_argument('--epoch', type=int, default=config.EPOCH)
parser.add_argument('--cropw', type=int, default=config.CROP_W)
parser.add_argument('--croph', type=int, default=config.CROP_H)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--lr', default=config.LEARNING_RATE)
parser.add_argument('--betas', default=config.BETAS)
parser.add_argument('--l2', default=config.L2_WEIGHT_DECAY)
parser.add_argument('--eps', default=config.EPS)

args = parser.parse_args()


if __name__ == '__main__':

    dataset = UVDataset(args.data, args.train, args.cropw, args.croph)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    model = Renderer(args.pyramidw, args.pyramidh)
    optimizer = Adam([
        {'params': model.texture.layer1.parameters(), 'weight_decay': args.l2[0]},
        {'params': model.texture.layer2.parameters(), 'weight_decay': args.l2[1]},
        {'params': model.texture.layer3.parameters(), 'weight_decay': args.l2[2]},
        {'params': model.texture.layer4.parameters(), 'weight_decay': args.l2[3]}],
        lr=args.lr, betas=args.betas, eps=args.eps)
    criterion = nn.L1Loss()

    print('Training started')
    for i in range(args.epoch):
        print('Epoch {}'.format(i))
        for imgs, uv_maps in dataloader:
            # TODO


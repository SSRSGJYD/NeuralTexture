import argparse
import numpy as np
import os
import random
import tensorboardX
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from dataset.uv_dataset import UVDataset
from model.pipeline import PipeLine

parser = argparse.ArgumentParser()
parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
parser.add_argument('--view_direction', type=bool, default=config.VIEW_DIRECTION)
parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
parser.add_argument('--checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
parser.add_argument('--logdir', type=str, default=config.LOG_DIR, help='directory to save checkpoint')
parser.add_argument('--train', default=config.TRAIN_SET)
parser.add_argument('--epoch', type=int, default=config.EPOCH)
parser.add_argument('--cropw', type=int, default=config.CROP_W)
parser.add_argument('--croph', type=int, default=config.CROP_H)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
parser.add_argument('--betas', type=str, default=config.BETAS)
parser.add_argument('--l2', type=str, default=config.L2_WEIGHT_DECAY)
parser.add_argument('--eps', type=float, default=config.EPS)
parser.add_argument('--load', type=str, default=config.LOAD)
parser.add_argument('--load_step', type=int, default=config.LOAD_STEP)
parser.add_argument('--epoch_per_checkpoint', type=int, default=config.EPOCH_PER_CHECKPOINT)
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, original_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 5:
        lr = original_lr * 0.2 * epoch
    elif epoch < 50:
        lr = original_lr
    elif epoch < 100:
        lr = 0.1 * original_lr
    else:
        lr = 0.01 * original_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    named_tuple = time.localtime()
    time_string = time.strftime("%m_%d_%Y_%H_%M", named_tuple)
    log_dir = os.path.join(args.logdir, time_string)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = tensorboardX.SummaryWriter(logdir=log_dir)

    checkpoint_dir = args.checkpoint + time_string
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    dataset = UVDataset(args.data, args.train, args.croph, args.cropw, args.view_direction)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    if args.load:
        print('Loading Saved Model')
        model = torch.load(os.path.join(args.checkpoint, args.load))
        step = args.load_step
    else:
        model = PipeLine(args.texturew, args.textureh, args.texture_dim, args.use_pyramid, args.view_direction)
        step = 0

    l2 = args.l2.split(',')
    l2 = [float(x) for x in l2]
    betas = args.betas.split(',')
    betas = [float(x) for x in betas]
    betas = tuple(betas)
    optimizer = Adam([
        {'params': model.texture.layer1, 'weight_decay': l2[0], 'lr': args.lr},
        {'params': model.texture.layer2, 'weight_decay': l2[1], 'lr': args.lr},
        {'params': model.texture.layer3, 'weight_decay': l2[2], 'lr': args.lr},
        {'params': model.texture.layer4, 'weight_decay': l2[3], 'lr': args.lr},
        {'params': model.unet.parameters(), 'lr': 0.1 * args.lr}],
        betas=betas, eps=args.eps)
    model = model.to('cuda')
    model.train()
    torch.set_grad_enabled(True)
    criterion = nn.L1Loss()

    print('Training started')
    for i in range(1, 1+args.epoch):
        print('Epoch {}'.format(i))
        adjust_learning_rate(optimizer, i, args.lr)
        for samples in dataloader:
            if args.view_direction:
                images, uv_maps, extrinsics, masks = samples
                # random scale
                scale = 2 ** random.randint(-1,1)
                images = F.interpolate(images, scale_factor=scale, mode='bilinear')
                
                uv_maps = uv_maps.permute(0, 3, 1, 2)
                uv_maps = F.interpolate(uv_maps, scale_factor=scale, mode='bilinear')
                uv_maps = uv_maps.permute(0, 2, 3, 1)

                step += images.shape[0]
                optimizer.zero_grad()
                RGB_texture, preds = model(uv_maps.cuda(), extrinsics.cuda())
            else:
                images, uv_maps, masks = samples
                # random scale
                scale = 2 ** random.randint(-1,1)
                images = F.interpolate(images, scale_factor=scale, mode='bilinear')
                uv_maps = uv_maps.permute(0, 3, 1, 2)
                uv_maps = F.interpolate(uv_maps, scale_factor=scale, mode='bilinear')
                uv_maps = uv_maps.permute(0, 2, 3, 1)
                
                step += images.shape[0]
                optimizer.zero_grad()
                RGB_texture, preds = model(uv_maps.cuda())

            loss1 = criterion(RGB_texture.cpu(), images)
            loss2 = criterion(preds.cpu(), images)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', loss.item(), step)
            print('loss at step {}: {}'.format(step, loss.item()))

        # save checkpoint
        if i % args.epoch_per_checkpoint == 0:
            print('Saving checkpoint')
            torch.save(model, args.checkpoint+time_string+'/epoch_{}.pt'.format(i))

if __name__ == '__main__':
    main()

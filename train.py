import argparse
import numpy as np
import os
import tensorboardX
import time
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
parser.add_argument('--pyramid_num', type=int, default=config.PYRAMID_NUM)
parser.add_argument('--view_direction', type=bool, default=config.VIEW_DIRECTION)
parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
parser.add_argument('--checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
parser.add_argument('--logdir', type=str, default=config.LOG_DIR, help='directory to save checkpoint')
parser.add_argument('--train', default=config.TRAIN_SET)
parser.add_argument('--epoch', type=int, default=config.EPOCH)
parser.add_argument('--cropw', type=int, default=config.CROP_W)
parser.add_argument('--croph', type=int, default=config.CROP_H)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--lr', default=config.LEARNING_RATE)
parser.add_argument('--betas', default=config.BETAS)
parser.add_argument('--l2', default=config.L2_WEIGHT_DECAY)
parser.add_argument('--eps', default=config.EPS)
parser.add_argument('--load', default=config.LOAD)
parser.add_argument('--load_step', type=int, default=config.LOAD_STEP)
parser.add_argument('--epoch_per_checkpoint', type=int, default=config.EPOCH_PER_CHECKPOINT)
args = parser.parse_args()


if __name__ == '__main__':

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
        model = Renderer(args.pyramidw, args.pyramidh, args.pyramid_num, args.view_direction)
        step = 0

    optimizer = Adam([
        {'params': model.texture.layer1, 'weight_decay': args.l2[0]},
        {'params': model.texture.layer2, 'weight_decay': args.l2[1]},
        {'params': model.texture.layer3, 'weight_decay': args.l2[2]},
        {'params': model.texture.layer4, 'weight_decay': args.l2[3]},
        {'params': model.unet.parameters()}],
        lr=args.lr, betas=args.betas, eps=args.eps)
    model = model.to('cuda')
    model.train()
    torch.set_grad_enabled(True)
    criterion = nn.L1Loss()

    print('Training started')
    for i in range(args.epoch):
        print('Epoch {}'.format(i+1))
        for samples in dataloader:
            if args.view_direction:
                images, uv_maps, sh_maps, masks = samples
                step += images.shape[0]
                optimizer.zero_grad()
                preds = model(uv_maps.cuda(), sh_maps.cuda()).cpu()
            else:
                images, uv_maps, masks = samples
                step += images.shape[0]
                optimizer.zero_grad()
                preds = model(uv_maps.cuda()).cpu()

            preds = torch.masked_select(preds, masks)
            images = torch.masked_select(images, masks)
            loss = criterion(preds, images)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', loss.item(), step)
            print('loss at step {}: {}'.format(step, loss.item()))

        # save checkpoint
        if (i+1) % args.epoch_per_checkpoint == 0:
            print('Saving checkpoint')
            torch.save(model, args.checkpoint+time_string+'/epoch_{}.pt'.format(i+1))

import argparse
import logging
import nni
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from dataset.uv_dataset import UVDataset
from model.renderer import Renderer

logger = logging.getLogger('neural_texture_AutoML')


def get_params():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--pyramidw', type=int, default=config.PYRAMID_W)
    parser.add_argument('--pyramidh', type=int, default=config.PYRAMID_H)
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
    args = parser.parse_args()
    return args


def main(args):

    # named_tuple = time.localtime()
    # time_string = time.strftime("%m_%d_%Y_%H_%M", named_tuple)
    # log_dir = os.path.join(args.logdir, time_string)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # writer = tensorboardX.SummaryWriter(logdir=log_dir)

    # checkpoint_dir = args.checkpoint + time_string
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)

    dataset = UVDataset(args.data, args.train, args.croph, args.cropw)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    if args.load:
        print('Loading Saved Model')
        model = torch.load(os.path.join(args.checkpoint, args.load))
        step = args.load_step
    else:
        model = Renderer(args.pyramidw, args.pyramidh)
        step = 0

    optimizer = Adam([
        {'params': model.texture.pyramid1.layer1, 'weight_decay': args.l2[0]},
        {'params': model.texture.pyramid2.layer1, 'weight_decay': args.l2[0]},
        {'params': model.texture.pyramid3.layer1, 'weight_decay': args.l2[0]},
        {'params': model.texture.pyramid1.layer2, 'weight_decay': args.l2[1]},
        {'params': model.texture.pyramid2.layer2, 'weight_decay': args.l2[1]},
        {'params': model.texture.pyramid3.layer2, 'weight_decay': args.l2[1]},
        {'params': model.texture.pyramid1.layer3, 'weight_decay': args.l2[2]},
        {'params': model.texture.pyramid2.layer3, 'weight_decay': args.l2[2]},
        {'params': model.texture.pyramid3.layer3, 'weight_decay': args.l2[2]},
        {'params': model.texture.pyramid1.layer4, 'weight_decay': args.l2[3]},
        {'params': model.texture.pyramid2.layer4, 'weight_decay': args.l2[3]},
        {'params': model.texture.pyramid3.layer4, 'weight_decay': args.l2[3]},
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
            images, uv_maps, masks = samples
            step += images.shape[0]
            optimizer.zero_grad()
            preds = model(uv_maps.cuda()).cpu()
            preds = torch.masked_select(preds, masks)
            images = torch.masked_select(images, masks)
            loss = criterion(preds, images)
            loss.backward()
            optimizer.step()
            nni.report_intermediate_result(loss.item())
            # writer.add_scalar('train/loss', loss.item(), step)
            print('loss at step {}: {}'.format(step, loss.item()))

        # save checkpoint
        # print('Saving checkpoint')
        # torch.save(model, args.checkpoint+time_string+'/epoch_{}.pt'.format(i+1))


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise

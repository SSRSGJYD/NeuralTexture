import argparse
import cv2
import numpy as np
import os
from skimage import img_as_ubyte
import sys
import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import config
from dataset.eval_dataset import EvalDataset
from model.pipeline import PipeLine

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
parser.add_argument('--test', default=config.TEST_SET, help='index list of test uv_maps')
parser.add_argument('--checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
parser.add_argument('--load', type=str, default=config.TEST_LOAD, help='checkpoint name')
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--save', type=str, default=config.SAVE_DIR, help='save directory')
parser.add_argument('--out_mode', type=str, default=config.OUT_MODE, choices=('video', 'image'))
parser.add_argument('--fps', type=int, default=config.FPS)
args = parser.parse_args()


if __name__ == '__main__':

    checkpoint_file = os.path.join(args.checkpoint, args.load)
    if not os.path.exists(checkpoint_file):
        print('checkpoint not exists!')
        sys.exit()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    dataset = EvalDataset(args.data, args.test, args.view_direction)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=EvalDataset.get_collect_fn(args.view_direction))

    model = torch.load(checkpoint_file)
    model = model.to('cuda')
    model.eval()
    torch.set_grad_enabled(False)

    if args.out_mode == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(os.path.join(args.save, 'render.mp4'), fourcc, 16,
                                     (dataset.width, dataset.height), True)
    print('Evaluating started')
    for samples in tqdm.tqdm(dataloader):
        if args.view_direction:
            uv_maps, extrinsics, masks, idxs = samples
            RGB_texture, preds = model(uv_maps.cuda(), extrinsics.cuda())
        else:
            uv_maps, masks, idxs = samples
            RGB_texture, preds = model(uv_maps.cuda())

        preds = preds.cpu()
        preds.masked_fill_(masks, 0) # fill invalid with 0

        # save result
        if args.out_mode == 'video':
            preds = preds.numpy()
            preds = np.clip(preds, -1.0, 1.0)
            for i in range(len(idxs)):
                image = img_as_ubyte(preds[i])
                image = np.transpose(image, (1,2,0))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                writer.write(image)
        else:
            for i in range(len(idxs)):
                image = transforms.ToPILImage()(preds[i])
                image.save(os.path.join(args.save, '{}_render.png'.format(idxs[i])))

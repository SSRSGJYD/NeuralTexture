import numpy as np
from PIL import Image
import random
import torch
import torchvision.transforms as transforms


def img_transform(image):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    image = image_transforms(image)
    return image


def mask_transform(mask):
    mask = torch.from_numpy(mask)
    return mask


def augment(img, mask, crop_size):
    '''
    :param img:  PIL input image
    :param mask: PIL input mask
    :param crop_size: a tuple (w, h)
    :return: image and mask
    '''
    # random mirror
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # random crop
    w, h = img.size
    crop_w, crop_h = crop_size
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)
    img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))

    # final transform
    img, mask = img_transform(img), mask_transform(mask)
    return img, mask



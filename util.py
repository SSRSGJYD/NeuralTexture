import numpy as np
from PIL import Image
import random
import torch
import torchvision.transforms as transforms


def img_transform(image):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    image = image_transforms(image)
    return image


def map_transform(map):
    map = torch.from_numpy(map)
    return map


def augment(img, map, crop_size):
    '''
    :param img:  PIL input image
    :param map: PIL input map
    :param crop_size: a tuple (w, h)
    :return: image and map
    '''
    # random mirror
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        map = map.transpose(Image.FLIP_LEFT_RIGHT)

    # random crop
    w, h = img.size
    crop_w, crop_h = crop_size
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)
    img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    map = map.crop((x1, y1, x1 + crop_w, y1 + crop_h))

    # final transform
    img, map = img_transform(img), map_transform(map)

    # mask for valid uv positions
    mask = map.ge(-1.0+1e-6)

    return img, map, mask



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
    :param map: numpy input map
    :param crop_size: a tuple (h, w)
    :return: image, map and mask
    '''
    # random mirror
    # if random.random() < 0.5:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #     map = np.fliplr(map)

    # random crop
    w, h = img.size
    crop_h, crop_w = crop_size
    w1 = random.randint(0, w - crop_w)
    h1 = random.randint(0, h - crop_h)
    img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
    map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]

    # final transform
    img, map = img_transform(img), map_transform(map)

    # mask for valid uv positions
    mask = torch.max(map, dim=2)[0].ge(-1.0+1e-6)
    mask = mask.repeat((3,1,1))

    return img, map, mask

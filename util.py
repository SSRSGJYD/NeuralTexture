import numpy as np
from PIL import Image
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def img_transform(image):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
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


# deprecated

# sh = np.zeros(9)
# sh[0] = 1 / np.sqrt(4 * np.pi)
# sh[1:4] = 2 * np.pi / 3 * np.sqrt(3 / (4 * np.pi))
# sh[4] = np.pi / 8 * np.sqrt(5 / (4 * np.pi))
# sh[5:8] = 3 * np.pi / 4 * np.sqrt(5 / (12 * np.pi))
# sh[8] = 3 * np.pi / 8 * np.sqrt(5 / (12 * np.pi))

# def view2sh(view_map, h, crop_h, w, crop_w):
#     '''
#     :param view_map: ndarray of (H, W, 3)
#     :param h: start position at height
#     :param crop_h:
#     :param w: start position at weight
#     :param crop_w:
#     :return: image, map and mask
#     '''
#     map = view_map[h:h+crop_h, w:w+crop_w, :]
#     sh_map = np.zeros((9, crop_h, crop_w), dtype=np.float32)
#     sh_map[0] = sh[0]
#     sh_map[1] = sh[1] * map[:, :, 2]
#     sh_map[2] = sh[2] * map[:, :, 1]
#     sh_map[3] = sh[3] * map[:, :, 0]
#     sh_map[4] = sh[4] * (2*map[:, :, 2]*map[:, :, 2]-map[:, :, 0]*map[:, :, 0]-map[:, :, 1]*map[:, :, 1])
#     sh_map[5] = sh[5] * map[:, :, 1] * map[:, :, 2]
#     sh_map[6] = sh[6] * map[:, :, 0] * map[:, :, 2]
#     sh_map[7] = sh[7] * map[:, :, 0] * map[:, :, 1]
#     sh_map[8] = sh[8] * (map[:, :, 0] * map[:, :, 0] - map[:, :, 1] * map[:, :, 1])
#     return sh_map


# def augment_view(img, map, view_map, crop_size):
#     '''
#     :param img:  PIL input image
#     :param map: numpy input map
#     :param view_map: numpy input map
#     :param crop_size: a tuple (h, w)
#     :return: image, map and mask
#     '''
#     # random mirror
#     # if random.random() < 0.5:
#     #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
#     #     map = np.fliplr(map)

#     # random crop
#     w, h = img.size
#     crop_h, crop_w = crop_size
#     w1 = random.randint(0, w - crop_w-1)
#     h1 = random.randint(0, h - crop_h-1)
#     img = img.crop((w1, h1, w1 + crop_w, h1 + crop_h))
#     map = map[h1:h1 + crop_h, w1:w1 + crop_w, :]
#     sh_map = view2sh(view_map, h1, crop_h, w1, crop_w)

#     # final transform
#     img, map, sh_map = img_transform(img), map_transform(map), map_transform(sh_map)
    
#     # mask for valid uv positions
#     mask = torch.max(map, dim=2)[0].ge(-1.0+1e-6)
#     mask = mask.repeat((3,1,1))

#     return img, map, sh_map, mask

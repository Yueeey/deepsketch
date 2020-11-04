"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import random


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def sparse_label(mask):
    """get sparse label from input mask

    Parameters:
        mask (PIL image) -- mask

    """
    # import pudb; pu.db
    mask = mask[0, :, :]
    H, W = mask.shape
    mean = [H/2, W/2]
    cov = np.diag([(H/2)**2, (W/2)**2])
    bg = np.zeros(mask.shape, dtype=np.float32)
    label = bg

    fore_num = np.random.geometric(p=0.125, size=1)
    back_num = np.random.geometric(p=0.125, size=1)
    num = fore_num + back_num
    fore = []
    back = []
    while (len(fore) + len(back)) < num:
        loc = np.random.multivariate_normal(mean, cov,1)
        while (loc[0, 0] > 255) or (loc[0, 1] > 255) or (loc[0, 0] < 0) or (loc[0, 1] < 0):
            loc = np.random.multivariate_normal(mean, cov,1)
        if mask[int(loc[0, 0]), int(loc[ 0,1])] == 1:
            if len(fore) < fore_num:
                fore.append(loc)
                for m in range(3):
                    m = m - 1
                    for n in range(3):
                        n = n - 1
                        x = int(loc[0, 0])+m
                        y = int(loc[0, 1])+n
                        if x > 255: x = 255
                        if y > 255: y = 255
                        label[x, y] = 1
        elif mask[int(loc[0, 0]), int(loc[ 0,1])] == -1:
            if len(back) < back_num:
                back.append(loc)
                for m in range(3):
                    m = m - 1
                    for n in range(3):
                        n = n - 1
                        x = int(loc[0, 0])+m
                        y = int(loc[0, 1])+n
                        if x > 255: x = 255
                        if y > 255: y = 255
                        label[x, y] = -1
        else:
            pass

    return label


    # mask_np = np.array(mask)
    # mask_np = mask_np[:,:,0]
    # loc_fore = []
    # loc_back = []

    # for i in mask_np.shape[0]:
    #     for j in mask_np.shape[1]:
    #         if mask_np ==



import ntpath
import os
import sys
import time
# from . import util, html
# from subprocess import Popen, PIPE
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score

# generation_path = Path('results/sketch2mask-sty-0/test_latest/images')
generation_path = Path('results/rebuttal/sketch2mask_16/test_latest/images')
# gt_path = Path('/vol/research/ycau/shapeNet/03001627')

threshold = 127

gen_dir = [x for x in generation_path.iterdir() if x.is_dir()]

iou_score = 0.
pre_score = 0.
rec_score = 0.
count = 0
for gen in tqdm(gen_dir):
    pred_path = gen.joinpath('fake_B_fore')
    gt_path = gen.joinpath('real_B')
    preds = sorted(pred_path.rglob("*.jpg"))
    gts = sorted(gt_path.rglob("*.jpg"))
    for i in range(len(preds)):
        pred = np.asarray(Image.open(str(preds[i])).convert('L'))
        gt = np.asarray(Image.open(str(gts[i])).convert('L'))
        # per_pixel_error = np.abs(gt - pred)
        pred_bn = (pred > threshold).astype(np.int)
        gt_bn = (gt > threshold).astype(np.int)

        # IoU score
        intersection = np.logical_and(gt_bn, pred_bn)
        union = np.logical_or(gt_bn, pred_bn)
        iou_score += np.sum(intersection) / np.sum(union)

        # Precision score
        pre_score += precision_score(gt_bn.flatten(), pred_bn.flatten(), average='binary')
        rec_score += recall_score(gt_bn.flatten(), pred_bn.flatten(), average='binary')
        count += 1

iou_score = iou_score / count
pre_score = pre_score / count
rec_score = rec_score / count
print('\nIoU score is: {:.3f}\n'.format(iou_score))
print('Precision score is: {:.3f}\n'.format(pre_score))
print('Recall score is: {:.3f}\n'.format(rec_score))



# def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
#     """Save images to the disk.

#     Parameters:
#         webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
#         visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
#         image_path (str)         -- the string is used to create image paths
#         aspect_ratio (float)     -- the aspect ratio of saved images
#         width (int)              -- the images will be resized to width x width

#     This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
#     """
#     image_dir = webpage.get_image_dir()
#     short_path = ntpath.basename(image_path[0])
#     name = os.path.splitext(short_path)[0]
#     model = ntpath.dirname(image_path[0]).split('/')[-3]
#     base_view=['azi_0_elev_10_0001', 'azi_45_elev_10_0001', 'azi_90_elev_10_0001', 'azi_135_elev_10_0001', 'azi_180_elev_10_0001', 'azi_225_elev_10_0001', 'azi_270_elev_10_0001', 'azi_315_elev_10_0001']

#     webpage.add_header(name)
#     ims, txts, links = [], [], []

#     for label, im_data in visuals.items():
#         im = util.tensor2im(im_data)
#         model_dir = os.path.join(image_dir, model, label)
#         base_dir = os.path.join(model_dir, 'base')
#         bias_dir = os.path.join(model_dir, 'bias')
#         image_name = '%s_%s.png' % (name, label)
#         if name in base_view:
#             if not os.path.exists(base_dir):
#                 os.makedirs(base_dir)
#             save_path = os.path.join(base_dir, short_path)
#         else:
#             if not os.path.exists(bias_dir):
#                 os.makedirs(bias_dir)
#             save_path = os.path.join(bias_dir, short_path)
#         # save_path = os.path.join(image_dir, image_name)
#         util.save_image(im, save_path, aspect_ratio=aspect_ratio)
#         ims.append(image_name)
#         txts.append(label)
#         links.append(image_name)
#     webpage.add_images(ims, txts, links, width=width)

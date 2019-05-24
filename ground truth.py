#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 19:52:25 2018

@author: yang
"""

import os
import path
import glob
import numpy as np
import PIL.Image
import PIL.ImageDraw

# ground truth directory
gt_text_dir = "./training_samples/ch4_training_localization_transcription_gt"

# original images directory
image_dir = "./training_samples/ch4_training_images/*.jpg"
imgDirs = []
imgLists = glob.glob(image_dir)

# where to save the images with ground truth boxes
imgs_save_dir = "./training_samples/ICDAR_with_GT"

for item in imgLists:
    imgDirs.append(item)

for img_dir in imgDirs:
    img = PIL.Image.open(img_dir)
    dr = PIL.ImageDraw.Draw(img)    

    img_basename = os.path.basename(img_dir)
    (img_name, temp2) = os.path.splitext(img_basename)
    # open the ground truth text file
    img_gt_text_name = "gt_" + img_name + ".txt"
    print (img_gt_text_name)

    bf = open(os.path.join(gt_text_dir, img_gt_text_name)).read().decode("utf-8-sig").encode("utf-8")#.decode("utf-8-sig").encode("utf-8").splitlines()

    for idx in bf:
        rect = []
        spt = idx.split(',')
        rect.append(np.float(spt[0]))
        rect.append(np.float(spt[1]))
        rect.append(np.float(spt[2]))
        rect.append(np.float(spt[3]))
        rect.append(np.float(spt[4]))
        rect.append(np.float(spt[5]))
        rect.append(np.float(spt[6]))
        rect.append(np.float(spt[7]))

        # draw the polygon with (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4)
        dr.polygon((rect[0], rect[1], rect[2], rect[3], rect[4], rect[5], rect[6], rect[7]), outline="red")

    img.save(os.path.join(imgs_save_dir, img_basename))
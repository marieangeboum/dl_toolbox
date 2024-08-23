#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:50:19 2022

@author: maboum
"""
import numpy as np
import os
import glob
from argparse import ArgumentParser
import time
import tabulate
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
import torch

parser = ArgumentParser()
parser.add_argument('--townA', type=str, default='tyrol-w')
args = parser.parse_args()
# temporaire  à changer pour mettre le nom de ville en args
image_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/{}*.tif'.format(args.townA)))
#image_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/*.tif'))

# list of E[X] for each channel/images
meanR = []
meanB = []
meanG = []

# list of E[X**2] for each channel/images
meanR2 = []
meanB2 = []
meanG2 = []

for image in image_paths_list : 
    image_read = rasterio.open(image)
    image_bands = image_read.read().astype(np.float32)
    meanR.append(image_bands[0].mean())
    meanG.append(image_bands[1].mean())
    meanB.append(image_bands[2].mean())
    meanR2.append((image_bands[0]**2).mean())
    meanG2.append((image_bands[1]**2).mean())
    meanB2.append((image_bands[2]**2).mean())

# 1/n*sum(E[X])
meanR_total=np.mean(np.array(meanR))
meanG_total=np.mean(np.array(meanG))
meanB_total=np.mean(np.array(meanB))

# 1/n*sum(E[X**2])   
meanR2_total=np.mean(np.array(meanR2))
meanG2_total=np.mean(np.array(meanG2))
meanB2_total=np.mean(np.array(meanB2))

# sigma = sqrt(E[X**2]-E[X]**2)
sigmaR = np.sqrt(meanR2_total - meanR_total**2)
sigmaG = np.sqrt(meanG2_total - meanG_total**2)
sigmaB = np.sqrt(meanB2_total - meanB_total**2)

mean = [meanR_total,meanG_total,meanB_total]
std = [sigmaR ,sigmaG ,sigmaB ]
print(mean, std, sep = "\n")
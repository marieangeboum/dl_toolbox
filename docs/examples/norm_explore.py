#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:38:22 2022

@author: Marie-Ange Boum
"""

import os
import glob
import numpy as np
import time
import tabulate
import cv2

import matplotlib
import matplotlib.pyplot as plt

import rasterio
from rasterio.windows import Window
from rasterio.plot import show_hist
from argparse import ArgumentParser

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms
from torchmetrics import Accuracy

import segmentation_models_pytorch as smp

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging

import dl_toolbox.inference as dl_inf
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.networks import UNet
from dl_toolbox.callbacks import SegmentationImagesVisualisation, CustomSwa, ConfMatLogger
from dl_toolbox.callbacks import plot_confusion_matrix, compute_conf_mat, EarlyStopping
from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_datasets import InriaDs, InriaAustinDs, InriaAllDs
from dl_toolbox.torch_datasets.utils import *

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

parser = ArgumentParser()
parser.add_argument('--townA', type=str, default='vienna')
parser.add_argument('--num_tile', type = str, default = '1')
args = parser.parse_args()



raster_path = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/{}{}.tif'.format(args.townA, args.num_tile)))
raster = rasterio.open(raster_path[0])
#image_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/{}*.tif'.format(args.townA)))
#image_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/*.tif'))

# R = []
# B = []
# G = []

# for image in image_paths_list : 
#     image_read = rasterio.open(image)
#     image_bands = image_read.read().astype(np.uint8)
 
# show_hist(raster, bins=500, lw=0.0, stacked=True, alpha=0.5,
#     histtype='stepfilled', title="Original bands ({})".format(args.townA))
# plt.savefig('hist_{}{}.png'.format(args.townA, args.num_tile))
# raster_norm_austin = (rasterio.open(raster_path[0]).read()- InriaAustinDs.stats['mean'].reshape(-1,1,1))/InriaAustinDs.stats['std'].reshape(-1,1,1)
# show_hist(raster_norm_austin, bins=500, lw=0.0, stacked=False, alpha=0.5,
#     histtype='stepfilled', title="Normalized bands ({})".format(args.townA))
# plt.savefig('hist_{}_norm_{}{}.png'.format(args.townA,args.townA, args.num_tile))

# raster_norm_all=(rasterio.open(raster_path[0]).read()- InriaAllDs.stats['mean'].reshape(-1,1,1))/InriaAllDs.stats['std'].reshape(-1,1,1)
# show_hist(raster_norm_all, bins=500, lw=0.0, stacked=False, alpha=0.5,
#     histtype='stepfilled', title="Normalized bands with Inria params ({})".format(args.townA))
# plt.savefig('hist_{}_norm_{}{}.png'.format('all',args.townA, args.num_tile))

raster_norm_all=(rasterio.open(raster_path[0]).read()- InriaAustinDs.stats['mean'].reshape(-1,1,1))/InriaAustinDs.stats['std'].reshape(-1,1,1)
show_hist(raster_norm_all, bins=500, lw=0.0, stacked=False, alpha=0.5,
    histtype='stepfilled', title="Normalized bands with Austin params ({})".format(args.townA))
plt.savefig('hist_{}_norm_{}{}.png'.format('austin',args.townA, args.num_tile))


# label_raster_path = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/{}{}.tif'.format(args.townA, args.num_tile)))
# label_raster = rasterio.open(label_raster_path[0]).read()/255
# show_hist(label_raster, bins=50, lw=0.0, stacked=False, alpha=0.3,
#     histtype='stepfilled', title="Normalized bands with Inria params ({})".format(args.townA, args.num_tile))
# plt.savefig('hist_label_{}{}.png'.format(args.townA, args.num_tile))


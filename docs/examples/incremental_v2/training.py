#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:03:39 2023

@author: maboum
"""
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import time
import tabulate
import shutil, sys 

import torch
import matplotlib.pyplot as plt
import matplotlib.image
import albumentations as A
from rasterio.windows import Window
from argparse import ArgumentParser
from sklearn.utils import shuffle
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
from dl_toolbox.torch_datasets import *
from dl_toolbox.torch_datasets.utils import *

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature



def main():   

    # parser for argument easier to launch .py file and inintalizing arg. correctly
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default = '/data/INRIA/AerialImageDataset/train')   
    parser.add_argument('--train_split_coef', type = float, default = 0.75)
    args = parser.parse_args()
    
    # Exécution sur GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    for data_source in ('austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna'):
        
        img = glob.glob(os.path.join(args.data_path, 'images/{}*.tif'.format(data_source)))
        lbl = glob.glob(os.path.join(args.data_path, 'gt/{}*.tif'.format(data_source)))
        
        img, lbl = shuffle(np.array(img),np.array(lbl))
        
        img = img.tolist()
        lbl = lbl.tolist()
        
        train_imgs = img[:int(len(img)*args.train_split_coef)]
        train_lbl = lbl[:int(len(lbl)*args.train_split_coef)]
        
        test_imgs = img[int(len(img)*args.train_split_coef):]
        test_lbl = lbl[int(len(lbl)*args.train_split_coef):]
        # Test dataset           
        # Get a list of all files in the source directory
        train_dir_imgs ='train_s1_images/'
        train_dir_lbl = 'train_s1_gt/'
        
        test_dir_imgs = 'test_s1_images/'
        test_dir_lbl = 'test_s1_gt/'
        
        
        if not os.path.exists(train_dir_imgs):
            os.makedirs(train_dir_imgs)
        if not os.path.exists(train_dir_lbl):
            os.makedirs(train_dir_lbl)
        if not os.path.exists(test_dir_imgs):
            os.makedirs(test_dir_imgs)
        if not os.path.exists(test_dir_lbl):
            os.makedirs(test_dir_lbl)
            
        # Copy each file from the source directory to the destination directory
        for img, lbl in zip(train_imgs, train_lbl):
            src_path_img = img
            src_path_lbl = lbl
            dst_path_img = os.path.join(train_dir_imgs, os.path.basename(img).split('/')[-1])
            dst_path_lbl = os.path.join(train_dir_lbl, os.path.basename(lbl).split('/')[-1])
            shutil.copy(src_path_img,  dst_path_img)
            shutil.copy(src_path_lbl,  dst_path_lbl)
        
        # Copy each file from the source directory to the destination directory
        for img, lbl in zip(test_imgs, test_lbl):
            src_path_img = img
            src_path_lbl = lbl
            dst_path_img = os.path.join(test_dir_imgs, os.path.basename(img).split('/')[-1])
            dst_path_lbl = os.path.join(test_dir_lbl, os.path.basename(lbl).split('/')[-1])
            shutil.copy(src_path_img,  dst_path_img)
            shutil.copy(src_path_lbl,  dst_path_lbl)    
        
           

if __name__ == "__main__":

    main()



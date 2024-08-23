#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:05:40 2023

@author: maboum
"""
import os
from dl_toolbox.torch_datasets import InriaDs
from rasterio.windows import Window, bounds, from_bounds, shape
dataset = InriaDs(
    image_path=os.path.join('/data/INRIA/AerialImageDataset/train', 'images/austin1.tif'),
    label_path=os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/austin1.tif'),
    crop_size=2500,
    crop_step=2500,
    img_aug='albu-brightness',
    tile=Window(col_off=0, row_off=0, width=5000, height=5000),
    fixed_crops=True)

trainset = ConcatDataset(train_datasets)

train_sampler = RandomSampler(
    data_source=trainset,
    replacement=False,
    num_samples=args.epoch_len)

train_dataloader = DataLoader(
    dataset=trainset,
    batch_size=args.sup_batch_size,
    sampler=train_sampler,
    collate_fn = CustomCollate(batch_aug = 'no'),
    num_workers=args.workers)


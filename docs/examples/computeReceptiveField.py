#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:10:55 2022

@author: https://gist.github.com/Nikasa1889/781a8eb20c5b32f8e378353cde4daa51#file-computereceptivefield-py
"""
import os
import glob
import math
import time
import tabulate
import cv2
import torch
import matplotlib.pyplot as plt
from rasterio.windows import Window
from argparse import ArgumentParser
from PIL import Image

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
from dl_toolbox.torch_datasets import InriaDs
from dl_toolbox.torch_datasets.utils import *

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature
from dl_toolbox.torch_receptive_field import receptive_field

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name='efficientnet-b0',
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_use_batchnorm=True)

model.load_state_dict(torch.load('smp_unet_vienna_FS1.pt'))
model.to(device)
model.eval()

#summary(model, (3,256,256))
convnet =   [[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0], [1, 1, 0]]
layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv', 'fc7-conv']
imsize = 227

conv_layer = model.encoder._conv_stem
# dataset = InriaDs(image_path = os.path.join('/data/INRIA/AerialImageDataset/train', 'images/chicago30.tif'), label_path = os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/vienna30.tif'), fixed_crops = True,
#                         tile=Window(col_off=0, row_off=0, width=5000, height=5000),
#                         crop_size=256,
#                         crop_step=256,
#                         img_aug='no')

#writer = SummaryWriter("INRIA -- smp_unet Vienna -- FS -- 1")
#writer = SummaryWriter("TEST")

#def main():

   # parser for argument easier to launch .py file and inintalizing arg. correctly
parser = ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./outputs")
parser.add_argument("--num_classes", type=int)
parser.add_argument("--train_with_void", action='store_true')
parser.add_argument("--eval_with_void", action='store_true')
parser.add_argument("--in_channels", type=int)
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--encoder", type=str)
parser.add_argument("--initial_lr", type=float)
parser.add_argument("--final_lr", type=float)
parser.add_argument("--lr_milestones", nargs=2, type=float)
parser.add_argument("--data_path", type=str)
parser.add_argument("--epoch_len", type=int, default=5000)
parser.add_argument("--sup_batch_size", type=int, default=16)
parser.add_argument("--crop_size", type=int, default=256)
parser.add_argument("--workers", default=6, type=int)
parser.add_argument('--img_aug', type=str, default='no')
parser.add_argument('--max_epochs', type=int, default=500)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
#parser.add_argument('--townA', type = str, default = 'austin')
args = parser.parse_args()

# execution sur GPU si  ce dernier est dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# Load pretrained model
model = smp.Unet(
    encoder_name='efficientnet-b0',
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_use_batchnorm=True)
model.load_state_dict(torch.load('smp_unet_vienna_FS1.pt'))
model.to(device)
model.eval()


# Loading data
townB_image_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/vienna*.tif'))
townB_label_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/vienna*.tif'))

img = Image.open(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/vienna30.tif'))
#preproc_img = preprocessing(img)
plt.imshow(img); plt.axis('off'); plt.show()
test_datasets = []
dataset = InriaDs(image_path = os.path.join('/data/INRIA/AerialImageDataset/train', 'images/vienna30.tif'), label_path = os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/vienna30.tif'), fixed_crops = True,
                        tile=Window(col_off=0, row_off=0, width=5000, height=5000),
                        crop_size=args.crop_size,
                        crop_step=args.crop_size,
                        img_aug=args.img_aug)
for data in dataset : 
    test_datasets.append(data)
    
testset = ConcatDataset(test_datasets)

test_dataloader = DataLoader(
    dataset=testset,
    shuffle=False,
    batch_size=1,
    collate_fn = CustomCollate(batch_aug = 'no'),
    num_workers=6)

preds = model(test_datasets[100]['image'].unsqueeze(0).cuda())
preds = (torch.sigmoid(preds))
preds_squeezed = preds[0,:,:].squeeze(0)
pred, idx = torch.max(preds_squeezed, dim = 0) # max pred along col axis 
col_max, col_idx = torch.max(pred, dim = 0)
pred, idx = torch.max(preds_squeezed, dim = 1)
row_max, row_idx = torch.max(pred, dim=0)


#Find the pixel with the maximum probability for the building class
# np_preds = preds.detach().squeeze(0).cpu().numpy()
# coord_max_activation= np.asarray(np.where(np_preds == np.max(np_preds, keepdims = True)))
# coord_max_activation = torch.from_numpy(coord_max_activation)# Use backprop to compute the receptive field

# load model
model_unet = smp.Unet(
    encoder_name='efficientnet-b0',
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_use_batchnorm=True)
# model is in train mode to be able to pass the gradients
model_unet.train()




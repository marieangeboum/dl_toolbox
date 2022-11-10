#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:28:35 2022

@author: maboum
"""
import os
import glob
from argparse import ArgumentParser

import segmentation_models_pytorch as smp
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window

from torchmetrics import Accuracy
import dl_toolbox.inference as dl_inf
from dl_toolbox.torch_datasets import InriaDs
from dl_toolbox.callbacks import SegmentationImagesVisualisation, CustomSwa, ConfMatLogger
from dl_toolbox.callbacks import plot_confusion_matrix, compute_conf_mat, EarlyStopping
from dl_toolbox.torch_collate import CustomCollate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = smp.Unet(
    encoder_name='efficientnet-b0',
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_use_batchnorm=True)
model.to(device)

model.load_state_dict(torch.load('smp_unet_vienna_FS1.pt'))
model.eval()

# temporaire  à changer pour mettre le nom de ville en args
townB_image_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/austin*.tif'))
townB_label_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/austin*.tif'))

coef = 0.2
test_img_paths_list = townB_image_paths_list[:int(len(townB_image_paths_list)*coef)]
test_lbl_paths_list = townB_label_paths_list[:int(len(townB_label_paths_list)*coef)]


writer = SummaryWriter("INRIA--smp_unet Vienna--epoch_len-{}sup_batch_size-{}max_epochs-{}_austintest".format(args.epoch_len,args.sup_batch_size,args.max_epochs))

# Test dataset
test_datasets = []
for image_path, label_path in zip(test_img_paths_list, test_lbl_paths_list):
    test_datasets.append(InriaDs(image_path = image_path, label_path = label_path, fixed_crops = True,
                        tile=Window(col_off=0, row_off=0, width=5000, height=5000),
                        crop_size=args.crop_size,
                        crop_step=args.crop_size,
                        img_aug=args.img_aug))
    
testset = ConcatDataset(test_datasets)

test_dataloader = DataLoader(
    dataset=testset,
    shuffle=False,
    batch_size=args.sup_batch_size,
    collate_fn = CustomCollate(batch_aug = 'no'),
    num_workers=args.workers)
# Since we are not training, we don't need to compute the gradients for the outputs
viz = SegmentationImagesVisualisation(writer = writer)
acc_sum = 0.0
with torch.no_grad():
    for i,batch_test in enumerate(test_dataloader) :
        images = batch_test['image'].to(device)
        target = batch_test['mask'].to(device)/255
        outputs = model(images)
        batch_test['preds'] = outputs
        viz.display_batch(writer, batch_test, i, 256, prefix ='test')
        cm = compute_conf_mat(
            torch.tensor(target).flatten().cpu(),
            torch.tensor((torch.sigmoid(outputs)>0.5).cpu().long().flatten().cpu()), 2)
        
        accuracy = Accuracy(num_classes=2).cuda()
        acc_sum += accuracy(torch.transpose(outputs,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
        metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy())   
        # if i% 25 == 0: 
        #     with open('metrics_valid_batch.txt', mode='w') as file_object:
        #         print()
        #         print(metrics_per_class_df, file=file_object)
        #         print(macro_average_metrics_df, file=file_object)
        #         print(micro_average_metrics_df, file=file_object)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:28:35 2022

@author: maboum
"""
import os
import cv2
import glob
from argparse import ArgumentParser
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window
from torchvision import datasets, transforms
from torchmetrics import Accuracy
import dl_toolbox.inference as dl_inf
from dl_toolbox.torch_datasets import InriaDs, InriaAustinDs, InriaViennaDs, InriaAllDs
from dl_toolbox.callbacks import SegmentationImagesVisualisation, CustomSwa, ConfMatLogger
from dl_toolbox.callbacks import plot_confusion_matrix, compute_conf_mat, EarlyStopping
from dl_toolbox.torch_collate import CustomCollate
import matplotlib.image
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
parser.add_argument('--townA', type = str, default = 'austin')
parser.add_argument('--townB', type = str, default = 'vienna')

args = parser.parse_args()
model = smp.Unet(
    encoder_name='efficientnet-b0',
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_use_batchnorm=True)
model.to(device)

model.load_state_dict(torch.load('smp_unet_austin_FS1.pt'))
model.eval()

# temporaire  à changer pour mettre le nom de ville en args
townB_image_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/{}*.tif'.format(args.townB)))
townB_label_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/{}*.tif'.format(args.townB)))

test_img_paths_list = townB_image_paths_list[:]
test_lbl_paths_list = townB_label_paths_list[:]

writer = SummaryWriter("inference_test-{}-to-{}_nonorm_all".format(args.townA,args.townB))
viz = SegmentationImagesVisualisation(writer = writer,freq = 1)
#writer = SummaryWriter('testinference')

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

grey_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3)])
norm_transforms = transforms.Compose([transforms.Normalize(InriaAllDs.stats['mean'], InriaAllDs.stats['std'])])

acc_sum = 0
IoU = 0

with torch.no_grad(): 
    for i, batch in enumerate(test_dataloader):
        
        image = batch['image'].to(device)
        target = (batch['mask']/255.).to(device)
        image = norm_transforms(image).to(device) 
        
        logits = model(image)
        batch['preds'] = logits
        batch['image'] = image
        
        cm = compute_conf_mat(
            torch.tensor(target).flatten().cpu(),
            torch.tensor((torch.sigmoid(logits)>0.5).cpu().long().flatten().cpu()), 2)
        
        metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy()) 
        IoU += metrics_per_class_df.IoU[1]
         
        accuracy = Accuracy(num_classes=2).cuda()
        acc_sum += accuracy(torch.transpose(logits,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
    test_acc = acc_sum/len(test_dataloader)
    IoU = IoU/len(test_dataloader)
    viz.display_batch(writer, batch, 10,i,prefix='test')
    print(test_acc)
    print(IoU)
            
# i=0
# good_metrics =[]
# bad_metrics  =[]

# for data in testset : 
#     data['image'] = data['image'].unsqueeze(0).to(device)
#     data['orig_image'] = data['orig_image'].unsqueeze(0)
#     data['orig_mask'] = data['orig_mask'].unsqueeze(0)
#     image_norm = norm_transforms(data['image'])
#     data['image'] = image_norm
#     data['mask'] = (data['mask'].unsqueeze(0)/255.).to(device)
#     target = data['mask']
#     output = model(image_norm)
#     data['preds'] = output
    
#     cm = compute_conf_mat(
#         torch.tensor(target).flatten().cpu(),
#         torch.tensor((torch.sigmoid(output)>0.5).cpu().long().flatten().cpu()), 2)
    
#     metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy()) 
    
#     data_stats = {'name' : 'crop_{}_num_{}'.format(args.townB, i),
#                   'image' : image_norm,'preds': (torch.sigmoid(output)>0.5).long(), 
#                   'macroRecall':macro_average_metrics_df.macroRecall[0],
#                   'macroPrecision': macro_average_metrics_df.macroPrecision[0], 
#                   'OAccuracy': macro_average_metrics_df.OAccuracy[0]}
    
    # if macro_average_metrics_df.macroRecall[0]>0.8 and macro_average_metrics_df.macroPrecision[0]>0.8 :
    #     good_metrics.append(data_stats)
        
        
    # elif macro_average_metrics_df.macroPrecision[0]<0.5 and macro_average_metrics_df.macroPrecision[0]<0.5 :
    #     bad_metrics.append(data_stats)
    #     img = data['orig_image'][0].cpu().numpy().transpose(1, 2, 0)/255
        
    #     matplotlib.image.imsave('crop_{}_num_{}.jpeg'.format(args.townB, i),img)
    #     with open('metrics_crop_{}_num_{}.txt'.format(args.townB, i), mode='w') as file_object:                    
    #         print(metrics_per_class_df, file=file_object)
    #         print(macro_average_metrics_df, file=file_object)
    #         print(micro_average_metrics_df, file=file_object)
    #     viz.display_batch(writer, data, 1,i,prefix='crop_{}_num_{}'.format(args.townB, i))
    # i+=1    


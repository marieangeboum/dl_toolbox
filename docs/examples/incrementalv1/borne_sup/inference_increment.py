#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:28:35 2022

@author: maboum
"""
import os
import cv2
import glob
import numpy as np
import pandas as pd
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
from dl_toolbox.torch_datasets import *
from dl_toolbox.callbacks import SegmentationImagesVisualisation, CustomSwa, ConfMatLogger
from dl_toolbox.callbacks import plot_confusion_matrix, compute_conf_mat, EarlyStopping
from dl_toolbox.torch_collate import CustomCollate
import matplotlib.image
from matplotlib import cm as cmap

def inference():

    # parser for argument easier to launch .py file and inintalizing arg. correctly
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_classes", type=int, default = 1)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--encoder", type=str, default = 'efficientnet-b0')
     
    parser.add_argument("--data_path", type=str, default = '/data/INRIA/AerialImageDataset/train')
   
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='no')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_source', type = str, default = 'austin')
    parser.add_argument('--inference_domain', type = str)
    parser.add_argument('--data_target', type = str, default = 'vienna')
    
    parser.add_argument('--tile_width', type = int, default = 5000)
    parser.add_argument('--tile_height', type = int, default = 5000)
    parser.add_argument('--model_path', type = str, default ="./models/unet_{}+{}_{}")
    
    # parser.add_argument('--test_imgs_paths_list', type= list)
    # parser.add_argument('--test_gt_paths_list', type =list)
    #parser.add_argument('--saved_model_path',type = str, default = ")
    args = parser.parse_args() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)    
    torch.autograd.set_detect_anomaly(True)
    
    inference_model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_use_batchnorm=True)
    
    inference_model.to(device)
    inference_model.load_state_dict(torch.load(args.model_path))
    inference_model.eval()
    inference_visu = (args.model_path+"_inference_{}").format(args.data_source,args.data_target, args.img_aug,args.inference_domain) 

    inference_writer = SummaryWriter(inference_visu)
    inference_seg_img_visu = SegmentationImagesVisualisation(writer = inference_writer,freq = 10)
       
    test_img_paths_list = ['/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin11.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin32.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin9.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin17.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin3.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin10.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin23.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin27.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin22.tif']
    test_lbl_paths_list = ['/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin11.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin32.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin9.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin17.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin3.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin10.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin23.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin27.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin22.tif']

    for image_path, label_path in zip(test_img_paths_list, test_lbl_paths_list):
        test_datasets.append(InriaDs(image_path = image_path, label_path = label_path, fixed_crops = True,
                            tile=Window(col_off=0, row_off=0, width=args.tile_width, height=args.tile_height),
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
      
    
    if args.data_target == 'austin':
        norm_transforms = transforms.Compose([transforms.Normalize(InriaAustinDs.stats['mean'], InriaAustinDs.stats['std'])])
    if args.data_target == 'vienna':
        norm_transforms = transforms.Compose([transforms.Normalize(InriaViennaDs.stats['mean'], InriaViennaDs.stats['std'])])
    if args.data_target == 'chicago':
        norm_transforms = transforms.Compose([transforms.Normalize(InriaAustinDs.stats['mean'], InriaChicagoDs.stats['std'])])
    if args.data_target == 'tyrol-w':
        norm_transforms = transforms.Compose([transforms.Normalize(InriaTyrolDs.stats['mean'], InriaTyrolDs.stats['std'])])
    if args.data_target == 'kitsap':
        norm_transforms = transforms.Compose([transforms.Normalize(InriaKitsapDs.stats['mean'], InriaKitsapDs.stats['std'])])
    
    acc_sum = 0.0
    iou     = 0.0
    precision = 0.0
    recall = 0.0
    
    crop_name, Recall, Precision, IoU = [],[],[],[]
    # accuracy = Accuracy(task = 'binary',num_classes=2).cuda()
    accuracy = Accuracy(num_classes=2).cuda()
    with torch.no_grad(): 
        for i, batch in enumerate(test_dataloader):            
            image = batch['image'].to(device)
            target = (batch['mask']/255.).to(device)
            image = norm_source_target(image).to(device) 
            output = inference_model(image)  
            batch['image'] = image            
            batch['preds'] = output.cpu()
            
            img = batch['orig_image'][0].cpu().numpy().transpose(1, 2, 0)/255
            
            cm = compute_conf_mat(
                    torch.tensor(target).flatten().cpu(),
                    torch.tensor((torch.sigmoid(output)>0.5).cpu().long().flatten().cpu()), 2)
            metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy()) 
            
            iou += metrics_per_class_df.IoU[1]
            precision += metrics_per_class_df.Precision[1]
            recall += metrics_per_class_df.Recall[1]
            acc_sum += accuracy(torch.transpose(output,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
            
            img = batch['orig_image'][0].cpu().numpy().transpose(1, 2, 0)/255
            if not os.path.exists('./{}'.format(args.data_target)):
                os.makedirs('./{}'.format(args.data_target))
            if os.path.isfile('./{}/orig_image_num_{}.jpeg'.format(args.data_target, i)):
                print('./{}/orig_image_num_{}.jpeg'.format(args.data_target, i),'already exists', sep = ' ')
            
            crop_name.append('orig_image_{}_num_{}'.format(args.data_target, i))
            Recall.append(metrics_per_class_df.Recall[1])
            Precision.append(metrics_per_class_df.Precision[1])
            IoU.append(metrics_per_class_df.IoU[1])
            matplotlib.image.imsave('./{}/orig_image_num_{}.jpeg'.format(args.data_target, i),img)
            inference_seg_img_visu.display_batch(inference_writer,batch, 1,i,prefix='orig_image_{}_num_{}'.format(args.data_target, i))   
        
        test_acc = acc_sum/ len(test_dataloader)   
        test_iou = iou/len(test_dataloader)
        test_precision = precision/len(test_dataloader)
        test_recall = recall/len(test_dataloader)
        
    metrics_name = ['Accuracy', 'IoU', 'Precision', 'Recall']    
    metrics_values = [test_acc,test_iou, test_precision, test_recall]
    dict_metrics = {'model' :args.model_path, 'metrics':metrics_name, 'values':metrics_values }  

    df_inference_metrics = pd.DataFrame({'crop_name' : crop_name, 'Recall' : Recall,'Precision ': Precision , 'IoU':IoU})
    df_global_inference = pd.DataFrame(dict_metrics)
    with pd.ExcelWriter('inference_metrics.xlsx', mode="a",  if_sheet_exists='overlay') as writer_xlsx:  
        df_inference_metrics.to_excel(writer_xlsx, sheet_name=inference_visu, index=False)
        df_global_inference.to_excel(writer_xlsx, sheet_name='global_{}'.format(inference_visu), index =False)

if __name__ == "__main__":
    inference()
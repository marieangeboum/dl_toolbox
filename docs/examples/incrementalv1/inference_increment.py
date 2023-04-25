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
    parser.add_argument("--sup_batch_size", type=int, default=1)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--used_seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_source', type = str)
    parser.add_argument('--inference_domain', type = str)
    parser.add_argument('--data_target', type = str, default = 'vienna')
    
    parser.add_argument('--tile_width', type = int, default = 5000)
    parser.add_argument('--tile_height', type = int, default = 5000)
    parser.add_argument('--model_path', type = str, default ="./models/unet_{}+{}_{}")
    parser.add_argument('--expe_name', type = str)
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
    inference_visu = "{}_{}{}_inf_{}".format(args.expe_name,args.data_source, args.data_target, args.inference_domain)
    print(inference_visu)
    inference_writer = SummaryWriter(inference_visu)
    inference_seg_img_visu = SegmentationImagesVisualisation(writer = inference_writer,freq = 10)
    
    test_datasets = []   
    
    if args.inference_domain == 'chicago' :            
        test_img_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/chicago21.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/chicago5.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/chicago10.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/chicago7.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/chicago19.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/chicago22.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/chicago29.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/chicago33.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/chicago6.tif']
        test_lbl_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/chicago21.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/chicago5.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/chicago10.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/chicago7.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/chicago19.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/chicago22.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/chicago29.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/chicago33.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/chicago6.tif']
    if args.inference_domain =='austin':
        test_img_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin11.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin32.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin9.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin17.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin3.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin10.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin23.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin27.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/austin22.tif']
        test_lbl_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin11.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin32.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin9.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin17.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin3.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin10.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin23.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin27.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/austin22.tif']
    if args.inference_domain == 'vienna':
        test_img_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/vienna7.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/vienna33.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/vienna34.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/vienna16.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/vienna27.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/vienna23.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/vienna21.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/vienna17.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/vienna30.tif']
        test_lbl_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/vienna7.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/vienna33.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/vienna34.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/vienna16.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/vienna27.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/vienna23.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/vienna21.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/vienna17.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/vienna30.tif']
    if args.inference_domain =='kitsap':
        test_img_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/kitsap21.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/kitsap16.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/kitsap5.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/kitsap4.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/kitsap19.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/kitsap30.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/kitsap8.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/kitsap2.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/kitsap28.tif']
        test_lbl_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/kitsap21.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/kitsap16.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/kitsap5.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/kitsap4.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/kitsap19.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/kitsap30.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/kitsap8.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/kitsap2.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/kitsap28.tif']
    if args.inference_domain == 'tyrol-w':
        test_img_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/tyrol-w32.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/tyrol-w16.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/tyrol-w31.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/tyrol-w17.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/tyrol-w4.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/tyrol-w29.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/tyrol-w27.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/tyrol-w33.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/images/tyrol-w26.tif']
        test_lbl_paths_list =['/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/tyrol-w32.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/tyrol-w16.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/tyrol-w31.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/tyrol-w17.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/tyrol-w4.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/tyrol-w29.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/tyrol-w27.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/tyrol-w33.tif', '/scratchf/DATASETS/INRIA/AerialImageDataset/train/gt/tyrol-w26.tif']
        
    
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
      
    norm_transforms = transforms.Compose([transforms.Normalize(InriaAllDs.stats['mean'], InriaAllDs.stats['std'])])
    # if args.inference_domain == 'vienna':
    # if args.inference_domain == 'austin':
    #     norm_transforms = transforms.Compose([transforms.Normalize(InriaAustinDs.stats['mean'], InriaAustinDs.stats['std'])])
    # if args.inference_domain == 'vienna':
    #     norm_transforms = transforms.Compose([transforms.Normalize(InriaViennaDs.stats['mean'], InriaViennaDs.stats['std'])])
    # if args.inference_domain == 'chicago':
    #     norm_transforms = transforms.Compose([transforms.Normalize(InriaAustinDs.stats['mean'], InriaChicagoDs.stats['std'])])
    # if args.inference_domain == 'tyrol-w':
    #     norm_transforms = transforms.Compose([transforms.Normalize(InriaTyrolDs.stats['mean'], InriaTyrolDs.stats['std'])])
    # if args.inference_domain == 'kitsap':
    #     norm_transforms = transforms.Compose([transforms.Normalize(InriaKitsapDs.stats['mean'], InriaKitsapDs.stats['std'])])
    
    acc_sum = 0.0
    iou     = 0.0
    precision = 0.0
    recall = 0.0
    
    crop_name, Recall, Precision, IoU = [],[],[],[]
    accuracy = Accuracy(task = 'binary',num_classes=2).cuda()
    #accuracy = Accuracy(num_classes=2).cuda()
    #args.data_target = args.inference_domain 
    with torch.no_grad(): 
        for i, batch in enumerate(test_dataloader):            
            image = batch['image'].to(device)
            target = (batch['mask']/255.).to(device)
            image =  norm_transforms(image).to(device) 
            
            output = inference_model(image)  
            batch['image'] = image            
            batch['preds'] = output.cpu()
            
            # img = batch['orig_image'][0].cpu().numpy().transpose(1, 2, 0)/255
            
            cm = compute_conf_mat(
                    torch.tensor(target).flatten().cpu(),
                    torch.tensor((torch.sigmoid(output)>0.5).cpu().long().flatten().cpu()), 2)
            metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy()) 
            
            iou += metrics_per_class_df.IoU[1]
            precision += metrics_per_class_df.Precision[1]
            recall += metrics_per_class_df.Recall[1]
            acc_sum += accuracy(torch.transpose(output,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
            
            crop_name.append('orig_image_{}_num_{}'.format(args.inference_domain, i))
            Recall.append(metrics_per_class_df.Recall[1])
            Precision.append(metrics_per_class_df.Precision[1])
            IoU.append(metrics_per_class_df.IoU[1])
            #matplotlib.image.imsave('./{}/orig_image_num_{}.jpeg'.format(args.data_target, i),img)
            #inference_seg_img_visu.display_batch(inference_writer,batch, 1,i,prefix='img_{}_num_{}'.format(args.inference_domain, i))   
        
        test_acc = acc_sum/ len(test_dataloader)   
        test_iou = iou/len(test_dataloader)
        test_precision = precision/len(test_dataloader)
        test_recall = recall/len(test_dataloader)
        
    metrics_name = ['IoU']    
    metrics_values = [test_iou]
    dict_metrics = {'model' :args.model_path, 'metrics':metrics_name, 'values':metrics_values, 'domains' :'{}_{}'.format(args.data_source,args.data_target), 'inference_domain': '{}'.format(args.inference_domain)}  

    df_inference_metrics = pd.DataFrame({'crop_name' : crop_name, 'Recall' : Recall,'Precision ': Precision , 'IoU':IoU})
    df_global_inference = pd.DataFrame(dict_metrics)
    # if os.path.isfile('inference_metrics_{}_{}_{}.xlsx'.format(args.data_source, args.data_target, args.inference_domain) ): 
    with pd.ExcelWriter('inference_oc_s{}.xlsx'.format(args.used_seed), mode="a",engine="openpyxl",  if_sheet_exists='overlay') as writer_xlsx:  
        #df_inference_metrics.to_excel(writer_xlsx, sheet_name="{}_inf_{}".format(args.expe_name, args.inference_domain), index=False)
        df_global_inference.to_excel(writer_xlsx, sheet_name="g_{}_inf_{}_s{}".format(args.expe_name, args.inference_domain, args.used_seed), index =False)
        print('OK')

if __name__ == "__main__":
    inference()
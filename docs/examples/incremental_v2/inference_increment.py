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


import openpyxl
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
    parser.add_argument('--inference_domain', type = str)
    parser.add_argument('--data_target', type = str, default = 'vienna')
    parser.add_argument('-n','--sequence_list',  nargs='+', default=[])
    parser.add_argument('--tile_width', type = int, default = 5000)
    parser.add_argument('--tile_height', type = int, default = 5000)
    parser.add_argument('--training_mode', type = str)
    parser.add_argument('--saved_model_path', type = str)
    parser.add_argument('--expe_name', type = str)
    
    args = parser.parse_args() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    
    df = []
    for seed in (544, 727) : 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)    
        torch.autograd.set_detect_anomaly(True)
        for args.training_mode in ("encoder_decoder", "decoder"):
            
            model_list = glob.glob('replay/models/s1/{}_*_{}_seed_{}'.format(args.training_mode, args.data_target, seed))
            for model in model_list : 
                
                inference_model = smp.Unet(                        
                    encoder_name=args.encoder,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                    decoder_use_batchnorm=True)
                
                inference_model.to(device)
                inference_model.load_state_dict(torch.load(model))
                inference_model.eval()
               
                for args.inference_domain in (args.sequence_list) : 
                    inference_visu = "{}_inf_{}".format(model ,args.inference_domain)
                    print(inference_visu)
                    inference_writer = SummaryWriter(inference_visu)
                    inference_seg_img_visu = SegmentationImagesVisualisation(writer = inference_writer,freq = 10)
                    print(args.inference_domain)
                    
                    test_datasets = []   
                    # args.data_path = '/d/maboum/DL/dl_toolbox/docs/examples/incremental_v2'
                    test_img_paths_list = glob.glob('test_s1_images/{}*.tif'.format(args.data_target))
                    test_lbl_paths_list = glob.glob('test_s1_gt/{}*.tif'.format(args.data_target))
                               
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
                            
                            # crop_name.append('orig_image_{}_num_{}'.format(args.inference_domain, i))
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
                    dict_metrics = {'model' :model, 'IoU':metrics_values, 'inference_domain': '{}'.format(args.inference_domain), 'expe' : args.training_mode, 'seed': seed}  
                    
                    # df_inference_metrics = pd.DataFrame({'crop_name' : crop_name, 'Recall' : Recall,'Precision ': Precision , 'IoU':IoU})
                    
                    df_global_inference = pd.DataFrame(dict_metrics)
                    # if os.path.isfile('inference_metrics_{}_{}_{}.xlsx'.format(args.data_source, args.data_target, args.inference_domain) ): 
                    df.append(df_global_inference) 
                    print(df_global_inference)
    df = pd.concat(df)
    with pd.ExcelWriter('inferences_s1.xlsx', mode='a', engine='openpyxl') as writer_xlsx:  
        #df_inference_metrics.to_excel(writer_xlsx, sheet_name="{}_inf_{}".format(args.expe_name, args.inference_domain), index=False)
        df.to_excel(writer_xlsx, sheet_name='{}_{}'.format(args.expe_name,len(args.sequence_list), index =False))
        print('OK')
                   

if __name__ == "__main__":
    inference()
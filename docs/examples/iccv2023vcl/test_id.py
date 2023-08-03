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
import fnmatch
import random
# import openpyxl
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
from dl_toolbox.networks import MultiHeadUnet
import matplotlib.image
from matplotlib import cm as cmap


parser = ArgumentParser()
parser.add_argument("--num_classes", type=int, default = 1)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--encoder", type=str, default = 'efficientnet-b2')     
parser.add_argument("--data_path", type=str, default = '/scratchf/CHALLENGE_IGN/FLAIR_1/train')   
parser.add_argument("--crop_size", type=int, default=512)
parser.add_argument("--workers", default=6, type=int)
parser.add_argument('--img_aug', type=str, default='no')
parser.add_argument('--sequence_path', type = str)
parser.add_argument("--sup_batch_size", type=int, default=8)
# parser.add_argument('-n','--sequence_list',  nargs='+', default=[])
parser.add_argument('--tile_width', type = int, default = 512)
parser.add_argument('--tile_height', type = int, default = 512)  
parser.add_argument('--train_split_coef', type = float, default = 0.7)  
parser.add_argument('--expe_name', type = str)   
parser.add_argument('--pretrain', type = str)
parser.add_argument('--num_gpu', type = str)

parser.add_argument('--len_seq', type = int, default=10)
args = parser.parse_args() 
os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu

def inference():   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = 1571  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True) 
    df_global = []
    df_local = []
    
    test_imgs = []
    sequence_list =  ['D072_2019','D031_2019', 'D023_2020', 'D009_2019', 'D014_2020', 'D007_2020', 'D004_2021', 'D080_2021', 'D055_2018', 'D035_2020']
    list_of_tuples = [(item, sequence_list.index(item)) for item in sequence_list]
    random.seed(seed) 
    for domain in sequence_list[:1] : 
        img = glob.glob(os.path.join(args.data_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
        random.shuffle(img)
        # train_imgs += img[:int(len(img)*args.train_split_coef)]
        # train_lbl  += lbl[:int(len(lbl)*args.train_split_coef)]
        test_imgs += img[int(len(img)*args.train_split_coef):]
        
    if args.pretrain == 'in':
        args.sequence_path ='in_sequence_{}/'
    if args.pretrain == 'flair1':
       args.sequence_path ='flair1_sequence_{}/' 
    if args.pretrain == 'inria':
        args.sequence_path ='inria_sequence_{}/' 
    if args.pretrain == 'no':
        args.sequence_path ='none_sequence_{}/' 
    
   
    def extract_elements(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
            elements = content.splitlines()
            return elements
    def filter_elements(pattern, my_list):
        return [element for element in my_list if pattern in element]

    def exclude_elements(exclusion_list, path_list):
        return [path for path in path_list if path not in exclusion_list]


    # element_list = filter_elements('CHALLENGE_IGN', paths)
    # Provide the path to your .txt file
    file_path = f'test_logfile{seed}.txt'
    print(file_path)
    # Call the function to extract the list of paths
    extracted_paths = extract_elements(file_path)
    pattern_imgs = 'CHALLENGE_IGN'
    test_imgs += filter_elements(pattern_imgs, extracted_paths)
    for args.expe_name in ('baseline', 'replay'):
    
        for step in range(len(sequence_list)) : 
            
            
            model_list = []
            if step == 0 : 
            
               model_list.extend(glob.glob(args.sequence_path.format(seed)+'/id_{}_step0model'.format(seed)))
               model_list = [element for element in model_list if 'res' not in element]     
            # for seed in seed_list : 
            else :
                
                model_list.extend(glob.glob(args.sequence_path.format(seed)+'/id_{}_{}_step{}*'.format(args.expe_name, seed, str(step))))  
                model_list = [element for element in model_list if 'res' not in element]        
            print(model_list)
            idx = step +1 
            
            for inference_domain in sequence_list[:idx]:   
                
                print(inference_domain)
                test_img_paths_list = [item for item in test_imgs if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/Z*_*/img/IMG_*.tif'.format(inference_domain)))]
                test_img_paths_list = test_img_paths_list[:150]
                # test_lbl_paths_list = [item for item in test_lbl if fnmatch.fnmatch(item, os.path.join(args.data_path, 'gt/{}*.tif'.format(inference_domain)))] 
                # test_img_paths_list = [os.path.join(args.data_path, 'images/{}22.tif'.format(inference_domain))]
                # test_lbl_paths_list = [os.path.join(args.data_path, 'gt/{}22.tif'.format(inference_domain))]
                test_datasets = [] 
                for img_path in test_img_paths_list : 
                    img_path_strings = img_path.split('/')
                    domain_pattern = img_path_strings[-4]
                    img_pattern = img_path_strings[-1].split('_')[-1].strip('.tif')
                    lbl_path = glob.glob(os.path.join(args.data_path, '{}/Z*_*/msk/MSK_{}.tif'.format(domain_pattern, img_pattern)))[0]
                    # for img_path, lbl_path in zip(domain_img_val, domain_lbl_val):             
                    test_datasets.append(FlairDs(image_path = img_path, label_path = lbl_path, fixed_crops = True,
                                        tile=Window(col_off=0, row_off=0,  width=args.tile_width, height=args.tile_height),
                                        crop_size=args.crop_size,
                                        crop_step=args.crop_size,
                                        img_aug=args.img_aug))
                
                testset =  ConcatDataset(test_datasets)
                test_dataloader = DataLoader(
                    dataset=testset,
                    shuffle=False,
                    batch_size=args.sup_batch_size,
                    collate_fn = CustomCollate(batch_aug = args.img_aug),
                    num_workers=args.workers)  
                norm_transforms = transforms.Compose([transforms.Normalize(InriaAllDs.stats['mean'], InriaAllDs.stats['std'])])

                for model in model_list :             
                    training_mode = model.split('step{}'.format(step))[1]
                    print(training_mode, step, args.expe_name,inference_domain, sep = ', ')            
                    model_name =  model.split('/')[-1]                 
                    inference_visu = args.sequence_path +"{}_{}_{}_{}".format(idx,args.expe_name,inference_domain, training_mode)
                    inference_writer = SummaryWriter(inference_visu)
                    inference_seg_img_visu = SegmentationImagesVisualisation(writer = inference_writer,freq = 1)              
                    
                    model_unet = smp.Unet(                        
                        encoder_name=args.encoder,
                        encoder_weights=None,
                        in_channels=3,
                        classes=1,
                        decoder_use_batchnorm=True)
                    # model_unet.to(device)
                    inference_model = MultiHeadUnet( model_unet, args.len_seq, head_idx = list_of_tuples[step][1])
                    inference_model.to(device)
                    inference_model.load_state_dict(torch.load(model))
                    
                    inference_model.eval()                   
                    acc_sum = 0.0
                    iou     = 0.0
                    precision = 0.0
                    recall = 0.0
                    crop_name, Recall, Precision, IoU, steps, mode = [],[],[],[],[],[]
                    accuracy = Accuracy(num_classes=2).cuda()
                    with torch.no_grad(): 
                        for i, batch in enumerate(test_dataloader):            
                            image = batch['image'][:,:args.in_channels,:,:].to(device)
                            target = (batch['mask']).to(device)
                            image =  norm_transforms(image).to(device)                     
                            output = inference_model(image)  
                            batch['image'] = image            
                            batch['preds'] = output.cpu()
                            
                            cm = compute_conf_mat(
                                    target.clone().detach().flatten().cpu(),
                                    (torch.sigmoid(output)>0.5).long().flatten().cpu().clone().detach(), 2)
                            metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy()) 
                                                    
                            # UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
                            iou += metrics_per_class_df.IoU[1]
                            precision += metrics_per_class_df.Precision[1]
                            recall += metrics_per_class_df.Recall[1]
                            acc_sum += accuracy(torch.transpose(output,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
                            
                            crop_name.append('{}_{}_num_{}'.format(inference_domain, training_mode, i))
                            Recall.append(metrics_per_class_df.Recall[1])
                            Precision.append(metrics_per_class_df.Precision[1])
                            IoU.append(metrics_per_class_df.IoU[1])
                            mode.append(model_name)
                            steps.append(idx)
                            inference_seg_img_visu.display_batch(inference_writer,batch, 1,i,prefix='{}_{}_num_{}'.format(inference_domain, training_mode, i))   
                        
                        test_acc = acc_sum/ len(test_dataloader)   
                        test_iou = iou/len(test_dataloader)
                        test_precision = precision/len(test_dataloader)
                        test_recall = recall/len(test_dataloader)
                        
                    metrics_name = ['IoU', 'precision', 'recall']    
                    metrics_values = [test_iou, test_precision, test_recall, test_acc.cpu()]
                    
                    inference_writer.add_scalar('Acc/val', test_acc, i)
                    inference_writer.add_figure('Confusion matrix', plot_confusion_matrix(cm.cpu(), class_names = ['0','1']),i)
                    inference_writer.add_scalar('IoU/val', test_iou,i)
                    inference_writer.add_scalar('Prec/val', test_precision,i)
                    inference_writer.add_scalar('Recall/val', test_recall, i)
                    dict_metrics = {'model' :model_name, 'IoU':[test_iou], 'Acc':[test_acc.cpu()], 'Precision':[test_precision], 'Recall':[test_recall], 'inference_domain': inference_domain, 'method' : training_mode, 'seed': seed,'step' : step}  
                    #df_inference_metrics = pd.DataFrame({'crop_name' : crop_name, 'Recall' : metrics_values[-1],'Precision ': metrics_values[1], 'IoU':metrics_values[0], 'model' :mode, 'method' : training_mode,'step':steps})
                    #df_local.append(df_inference_metrics)                
                    df_global_inference = pd.DataFrame(dict_metrics)
                    df_global.append(df_global_inference) 
                    print(len(df_global))   
                    # print(df_global_inference)
        df_global_inferences = pd.concat(df_global)
        # df_local = pd.concat(df_local)
    with pd.ExcelWriter('test_id_1571.xlsx', mode='a', engine='openpyxl') as writer_xlsx:  
        # df_local.to_excel(writer_xlsx, sheet_name="crops_{}".format(args.expe_name), index=False)
        df_global_inferences.to_excel(writer_xlsx, sheet_name='{}_{}'.format(args.pretrain,args.expe_name), index =False)
    
                   

if __name__ == "__main__":
    inference()
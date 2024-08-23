#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:09:03 2023

@author: maboum
"""
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import time
import tabulate

import torch
import matplotlib.pyplot as plt
import matplotlib.image
import albumentations as A
from rasterio.windows import Window
from argparse import ArgumentParser

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
from dl_toolbox.torch_datasets import InriaDs, InriaAustinDs, InriaAllDs, InriaViennaDs
from dl_toolbox.torch_datasets.utils import *

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature


writer = SummaryWriter()
def main():     
    
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_classes", type=int, default = 1)
    parser.add_argument("--train_with_void", action='store_true')
    parser.add_argument("--eval_with_void", action='store_true')
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--encoder", type=str, default = 'efficientnet-b0')
    parser.add_argument("--initial_lr", type=float, default = 0.01)
    parser.add_argument("--final_lr", type=float, default = 0.005)
    parser.add_argument("--lr_milestones", nargs=2, type=float, default=(20,80))
    parser.add_argument("--data_path", type=str, default = '/data/INRIA/AerialImageDataset/train')
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='no')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_source', type = str, default = 'austin')
    parser.add_argument('--data_target', type = str, default = 'vienna')
    parser.add_argument('--split_coef', type = float, default = 0.8)
    parser.add_argument('--tile_width', type = int, default = 5000)
    parser.add_argument('--tile_height', type = int, default = 5000)
    parser.add_argument('--model_path', type = str, default ="smp_unet_{}_{}_ft_{}")
    parser.add_argument('--saved_model_path', type = str, default ="smp_unet_{}_{}")
    args = parser.parse_args()
    # execution sur GPU si  ce dernier est dispo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.autograd.set_detect_anomaly(True) # stops training if something is wrong
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Tensorboard visualization
    model_result_viz = (args.model_path+"_result_viz").format(args.data_source, args.data_target, args.data_target)   
    writer = SummaryWriter(model_result_viz)
    
    # Data Augmentation 
    if args.data_target == 'austin':
        norm_transforms = transforms.Compose([transforms.Normalize(InriaAustinDs.stats['mean'], InriaAustinDs.stats['std'])])
    if args.data_target == 'vienna':
        norm_transforms = transforms.Compose([transforms.Normalize(InriaViennaDs.stats['mean'], InriaViennaDs.stats['std'])])
    
    # temporaire  à changer pour mettre le nom de ville en args
    townA_image_paths_list = glob.glob(os.path.join(args.data_path, 'images/{}*.tif'.format(args.data_target)))
    townA_label_paths_list = glob.glob(os.path.join(args.data_path, 'gt/{}*.tif'.format(args.data_target)))
    
   
    train_img_paths_list = townA_image_paths_list[:int(len(townA_image_paths_list)*args.split_coef)]
    train_lbl_paths_list = townA_label_paths_list[:int(len(townA_label_paths_list)*args.split_coef)]
    
    val_img_paths_list = townA_image_paths_list[int(len(townA_image_paths_list)*args.split_coef):]
    val_lbl_paths_list = townA_label_paths_list[int(len(townA_label_paths_list)*args.split_coef):]
            
    # Train dataset
    train_datasets = []    
    for image_path, label_path in zip(train_img_paths_list, train_lbl_paths_list):
        train_datasets.append(InriaDs(image_path = image_path, label_path = label_path, fixed_crops = False,
                            tile=Window(col_off=0, row_off=0, width=5000, height=5000),
                            crop_size=args.crop_size,
                            crop_step=args.crop_size,
                            img_aug=args.img_aug))
    trainset = ConcatDataset(train_datasets)
    
    # Validation dataset
    val_datasets = []
    for image_path, label_path in zip(val_img_paths_list, val_lbl_paths_list):
         val_datasets.append(InriaDs(image_path = image_path, label_path = label_path, fixed_crops = True,
                            tile=Window(col_off=0, row_off=0, width=5000, height=5000),
                            crop_size=args.crop_size,
                            crop_step=args.crop_size,
                        img_aug=args.img_aug))  
         
    valset =  ConcatDataset(val_datasets)
    
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
    #print(len(train_dataloader.dataset))
    val_dataloader = DataLoader(
        dataset=valset,
        shuffle=False,
        batch_size=args.sup_batch_size,
        collate_fn = CustomCollate(batch_aug = 'no'),
        num_workers=args.workers)
    #print(len(val_dataloader.dataset))
    
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_use_batchnorm=True)
    model.to(device)
    model.load_state_dict(torch.load(args.saved_model_path.format(args.data_source, args.img_aug)))
    
    for param in model.encoder.parameters():
        #param.requires_grad = False 
        print(param.requires_grad)
        
    # A changer pour le masking ("reduction=none")
    # initializing loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    
    # initializing the optimizer
    optimizer = SGD(
        model.parameters(),
        lr=args.initial_lr,
        momentum=0.9)
    
    # definition of learning rate
    def lambda_lr(epoch):
    
        m = epoch / args.max_epochs
        if m < args.lr_milestones[0]:
            return 1
        elif m < args.lr_milestones[1]:
            return 1 + ((m - args.lr_milestones[0]) / (
                        args.lr_milestones[1] - args.lr_milestones[0])) * (
                               args.final_lr / args.initial_lr - 1)
        else:
            return args.final_lr / args.initial_lr
    
    
    scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)

    # Callbacks définitions
    viz = SegmentationImagesVisualisation(writer = writer,freq = 10)
    columns = ['ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time']
    early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=args.model_path.format(args.data_source, args.img_aug,args.data_target))
    crop_name,F1, Recall, Precision, IoU, macroRecall, macroPrecision, crop_path = [],[],[],[],[],[],[],[]
    
    accuracy = Accuracy(task = 'binary',num_classes=2).cuda()
    
    start_epoch = 0
    storage = False
    min_loss = 0
    for epoch in range(start_epoch, args.max_epochs):
    
        time_ep = time.time()
        loss_sum = 0.0
        acc_sum = 0.0
        
        model.train()
    
        for i, batch in enumerate(train_dataloader):
    
            image = batch['image'].to(device)
            target = (batch['mask']/255.).to(device)
            image = norm_transforms(image).to(device)
            # à éviter le faire qd les images sont encore sur CPU
            # image = torch.tensor(pixel_transform(image = image.cpu().detach().numpy())['image']).to(device)# à faire avant !!!
            
            # clear gradients wrt parameters
            optimizer.zero_grad()           
            # forward to get outputs
            logits = model(image)            
            # calculate loss
            loss = loss_fn(logits, target)
            batch['preds'] = logits
            batch['image'] = image
            # getting gradients wrt parameters
            loss.backward()
            # updating parameters
            optimizer.step()            
            acc_sum += accuracy(torch.transpose(logits,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
            loss_sum += loss.item()
            
        
        viz.display_batch(writer, batch, 5,epoch,prefix='train')
        train_loss = {'loss': loss_sum / len(train_dataloader)}
        train_acc = {'acc': acc_sum/len(train_dataloader)}
        writer.add_scalar('Loss/train', train_loss['loss'], epoch+1)
        writer.add_scalar('Acc/train', train_acc['acc'], epoch+1)
        
        # storage = True if abs(min_loss - train_loss['loss']) < 0.01 else False
        # min_loss = train_loss['loss']
            
        loss_sum = 0.0
        acc_sum = 0.0
        iou = 0.0
        precision = 0.0
        recall = 0.0
        
        scheduler.step()
        model.eval()
        
        for i, batch in enumerate(val_dataloader):
    
            image = batch['image'].to(device)
            target = (batch['mask']/255.).to(device)           
            image = norm_transforms(image).to(device)
                        
            output = model(image) 
            loss = loss_fn(output, target) 
            
            batch['preds'] = output
            batch['image'] = image
            
            cm = compute_conf_mat(
                target.clone().detach().flatten().cpu(),
                ((torch.sigmoid(output)>0.5).cpu().long().flatten()).clone().detach(), 2)
            metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy()) 
            
            iou += metrics_per_class_df.IoU[1]
            precision += metrics_per_class_df.Precision[1] 
            recall += metrics_per_class_df.Recall[1]
            loss_sum += loss.item()
            acc_sum += accuracy(torch.transpose(output,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
            
            # if storage : 
            #     crop_name.append('orig_image_{}_num_{}'.format(args.data_source, i))
                # img_orig = batch['orig_image'][0].cpu().numpy().transpose(1, 2, 0)/255.
                # img = batch['image'][0].cpu().numpy().transpose(1, 2, 0)
                # if os.path.isfile('./{}/orig_image_{}_num_{}.jpeg'.format(model_result_viz,args.data_source, i)):
                #     print('./{}/orig_image_{}_num_{}.jpeg'.format(model_result_viz,args.data_source, i) + ' already exists')
                # if os.path.isfile('./{}/image_{}_num_{}.jpeg'.format(model_result_viz,args.data_source, i)):
                #     print('./{}/image_{}_num_{}.jpeg'.format(model_result_viz,args.data_source, i) + ' already exists')
                # else: 
                #     try : 
                #         matplotlib.image.imsave('./{}/orig_image_{}_num_{}.jpeg'.format(model_result_viz,args.data_source, i),img_orig)
                #         #matplotlib.image.imsave('./{}/image_{}_num_{}.jpeg'.format(model_result_viz,args.data_source, i),img)
                #     except ValueError : 
                #         print('Could not save that image')     
                # Recall.append(metrics_per_class_df.Recall[1])
                # Precision.append(metrics_per_class_df.Precision[1])
                # IoU.append(metrics_per_class_df.IoU[1])               
                    
        viz.display_batch(writer, batch,5, epoch,prefix = 'val') 
        val_iou = {'iou':iou/len(val_dataloader)}
        val_precision = {'prec':precision/len(val_dataloader)}
        val_recall = {'recall':recall/len(val_dataloader)}
        val_loss = {'loss': loss_sum / len(val_dataloader)} #
        val_acc = {'acc': acc_sum/ len(val_dataloader)}
        
        early_stopping(val_loss['loss'], model)
        if early_stopping.early_stop:
                print("Early stopping")
                break
            
        time_ep = time.time() - time_ep
        writer.add_scalar('Loss/val', val_loss['loss'], epoch+1)
        writer.add_scalar('Acc/val', val_acc['acc'], epoch+1)
        writer.add_figure('Confusion matrix', plot_confusion_matrix(cm.cpu(), class_names = ['0','1']), epoch+1)
        writer.add_scalar('IoU/val', val_iou['iou'], epoch+1)
        writer.add_scalar('Prec/val', val_precision['prec'], epoch+1)
        writer.add_scalar('Recall/val', val_recall['recall'], epoch+1)
               
        values = [epoch + 1, train_loss['loss'], val_loss['loss'],train_acc['acc'],val_acc['acc'], time_ep]
        table = tabulate.tabulate([values], columns, tablefmt='simple',
                                  floatfmt='8.4f')
        print(table)
        # perf_df = pd.DataFrame({'crop_name' : crop_name, 'Recall' : Recall,'Precision ': Precision , 'IoU':IoU })
        # if os.path.isfile((args.model_path +'_metrics.xlsx').format(args.data_source)):
        #     os.remove((args.model_path +'_metrics.xlsx').format(args.data_source))
        # with pd.ExcelWriter((args.model_path +'_metrics.xlsx').format(args.data_source), mode="w") as writer_xlsx:  
        #     perf_df.to_excel(writer_xlsx, sheet_name=model_result_viz, index=False)
        with open( model_result_viz+'.txt', mode = 'w')  as f : 
            f.write("val_acc : "+str(np.float64(val_acc['acc']))+'\n\n')
            f.write("val_iou :"+str(np.float64(val_iou['iou']))+'\n\n')
            f.write("val_precision "+str(np.float64(val_precision['prec']))+'\n\n')
            f.write("val_recall "+str(np.float64(val_recall['recall']))+'\n\n')
    writer.add_graph(model, image)
    writer.flush()
    writer.close()
    
if __name__ == "__main__":

    main()
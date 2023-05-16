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
    # parser.add_argument("--data_path", type=str, default = '/data/INRIA/AerialImageDataset/train')
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='no')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_source', type = str, default = 'austin')
    parser.add_argument('--prev', type = str, default = 'austin')
    parser.add_argument('--data_target', type = str, default = 'vienna')
    parser.add_argument('--train_split_coef', type = float, default = 0.8)
    parser.add_argument('--tile_width', type = int, default = 5000)
    parser.add_argument('--tile_height', type = int, default = 5000)
    parser.add_argument('--model_path', type = str, default ="baseline/models/s1/bl_{}_{}_ft_{}_seed_{}")
    parser.add_argument('--saved_model_path',type = str)
    parser.add_argument('--training_mode', type = str)
    # parser.add_argument('--expe_name', type = str, default = "bl")
    parser.add_argument('--run', type = int)
    parser.add_argument('--encoder_weights', type = str, default = "imagenet")
    args = parser.parse_args()
    
    # Exécution sur GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.saved_model_path = 'bl_{}_chicago_ft_{}_seed_{}'.format(args.training_mode, args.data_source,args.run)
    args.saved_model_path = 'baseline/models/s1/bl_{}_{}_ft_{}_seed_{}'.format(args.training_mode,args.prev, args.data_source,args.run)
            
    seed = int((args.saved_model_path).split('_')[-1])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True) # stops training if something is wrong
    
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
    
    train_visu = (args.model_path+"_results").format(args.training_mode,args.data_source, args.data_target, seed)   
    train_writer = SummaryWriter(train_visu)    
    seg_img_visu = SegmentationImagesVisualisation(writer = train_writer,freq = 10)
    
    # Tiles Domain B
    b_domain_img = glob.glob(os.path.join('train_s1_images/{}*.tif'.format(args.data_target)))
    b_domain_lbl = glob.glob(os.path.join('train_s1_gt/{}*.tif'.format(args.data_target)))
    b_domain_img, b_domain_lbl = shuffle(np.array(b_domain_img),np.array(b_domain_lbl))
    b_domain_img = b_domain_img.tolist()
    b_domain_lbl = b_domain_lbl.tolist()
    b_domain_img_train = b_domain_img[:int(len(b_domain_img)*args.train_split_coef)]
    b_domain_lbl_train = b_domain_lbl[:int(len(b_domain_img)*args.train_split_coef)]       
    b_domain_img_val = b_domain_img[int(len(b_domain_img)*args.train_split_coef):]
    b_domain_lbl_val = b_domain_lbl[int(len(b_domain_img)*args.train_split_coef):]
    
    # Train dataset
    train_datasets = []
    
    for image_path, label_path in zip(b_domain_img_train, b_domain_lbl_train):
        train_datasets.append(InriaDs(image_path = image_path, label_path = label_path, fixed_crops = False,
                            tile=Window(col_off=0, row_off=0, width=args.tile_width, height=args.tile_height),
                            crop_size=args.crop_size,
                            crop_step=args.crop_size,
                            img_aug=args.img_aug))
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
    
    # Validation dataset
    val_datasets = []
    for image_path, label_path in zip(b_domain_img_val , b_domain_lbl_val):
         val_datasets.append(InriaDs(image_path = image_path, label_path = label_path, fixed_crops = True,
                            tile=Window(col_off=0, row_off=0, width=args.tile_width, height=args.tile_height),
                            crop_size=args.crop_size,
                            crop_step=args.crop_size,
                            img_aug=args.img_aug))  
    valset =  ConcatDataset(val_datasets) 
    val_dataloader = DataLoader(
        dataset=valset,
        shuffle=False,
        batch_size=args.sup_batch_size,
        collate_fn = CustomCollate(batch_aug = 'no'),
        num_workers=args.workers)

    # UNet SS task
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
        decoder_use_batchnorm=True)
    model.to(device)
    model.load_state_dict(torch.load(args.saved_model_path))
    
    # Initialize Loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    if args.training_mode == "decoder":
        for param in model.encoder.parameters():
            param.requires_grad = False 
        # initializing the optimizer
        optimizer = SGD(
            model.decoder.parameters(),
            lr=args.initial_lr,
            momentum=0.9)
    elif args.training_mode == "encoder_decoder":
        # initializing the optimizer
        optimizer = SGD(
            model.parameters(),
            lr=args.initial_lr,
            momentum=0.9)
        
    # Learning rate 
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
    start_epoch = 0    
    accuracy = Accuracy(task = 'binary',num_classes=2).cuda()
    columns = ['ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time']
    early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.001,path=args.model_path.format(args.training_mode,args.data_source, args.data_target,seed))  
    for epoch in range(start_epoch, args.max_epochs):
        loss_sum = 0.0
        acc_sum  = 0.0
        time_ep = time.time()
        model.train()
        
        for i, batch in enumerate(train_dataloader):    
            image = batch['image'].to(device)
            target = (batch['mask']/255.).to(device)
            image = norm_transforms(image).to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad() 
            # Forward
            logits = model(image)            
            loss = loss_fn(logits, target)
            batch['preds'] = logits
            batch['image'] = image
            # Getting gradients w.r.t. parameters
            loss.backward()
            optimizer.step()  # updating parameters    
            acc_sum += accuracy(torch.transpose(logits,0,1).reshape(2, -1).t(), 
                                torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
            loss_sum += loss.item()
        seg_img_visu.display_batch(train_writer, batch, 10,epoch,prefix='train')
        train_loss = {'loss': loss_sum / len(train_dataloader)}
        train_acc = {'acc': acc_sum/len(train_dataloader)}
        train_writer.add_scalar('Loss/train', train_loss['loss'], epoch+1)
        train_writer.add_scalar('Acc/train', train_acc['acc'], epoch+1)
               
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
            acc_sum += accuracy(torch.transpose(output,0,1).reshape(2, -1).t(), 
                                torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
                    
        seg_img_visu.display_batch(train_writer, batch,15, epoch,prefix = 'val') 
        val_iou = {'iou':iou/len(val_dataloader)}
        val_precision = {'prec':precision/len(val_dataloader)}
        val_recall = {'recall':recall/len(val_dataloader)}
        val_loss = {'loss': loss_sum / len(val_dataloader)} 
        val_acc = {'acc': acc_sum/ len(val_dataloader)}
        
        early_stopping(val_loss['loss'], model)
        if early_stopping.early_stop:
                print("Early stopping")
                break
        time_ep = time.time() - time_ep
        train_writer.add_scalar('Loss/val', val_loss['loss'], epoch+1)
        train_writer.add_scalar('Acc/val', val_acc['acc'], epoch+1)
        train_writer.add_figure('Confusion matrix', plot_confusion_matrix(cm.cpu(), class_names = ['0','1']), epoch+1)
        train_writer.add_scalar('IoU/val', val_iou['iou'], epoch+1)
        train_writer.add_scalar('Prec/val', val_precision['prec'], epoch+1)
        train_writer.add_scalar('Recall/val', val_recall['recall'], epoch+1)
        values = [epoch + 1, train_loss['loss'], val_loss['loss'],train_acc['acc'],val_acc['acc'], time_ep]
        table = tabulate.tabulate([values], columns, tablefmt='simple',
                                  floatfmt='8.4f')
        print(table)
        
    train_writer.add_graph(model, image)
    train_writer.flush()
    train_writer.close()
    f=open("logfile.txt","a+")
    txt = "Seed: {} \n baseline {} domaine A {} domaine B {} \n\n".format(seed, args.training_mode,args.data_source, args.data_target)
    f.write(txt)
    f.close()   

if __name__ == "__main__":

    main()

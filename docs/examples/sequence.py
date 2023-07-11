# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:45:36 2023

@author: maboum
"""

import os
import glob
import numpy as np
import time
import tabulate
import torch
import fnmatch
import random
import rasterio

import dl_toolbox.inference as dl_inf
import segmentation_models_pytorch as smp

from rasterio.windows import Window
from argparse import ArgumentParser
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms
from torchmetrics import Accuracy

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.networks import UNet
from dl_toolbox.callbacks import SegmentationImagesVisualisation, CustomSwa, ConfMatLogger
from dl_toolbox.callbacks import plot_confusion_matrix, compute_conf_mat, EarlyStopping
from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_datasets import *
from dl_toolbox.torch_datasets.utils import *


parser = ArgumentParser()

parser.add_argument("--train_with_void", action='store_true')
parser.add_argument("--eval_with_void", action='store_true')
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--output_dir", type=str, default="./outputs")

parser.add_argument("--in_channels", type=int, default=3)

parser.add_argument("--num_classes", type=int, default = 1)    
parser.add_argument("--initial_lr", type=float, default = 0.01)
parser.add_argument("--final_lr", type=float, default = 0.005)
parser.add_argument("--lr_milestones", nargs=2, type=float, default=(20,80))
parser.add_argument("--data_path", type=str, default ='D:/maboum/flair_merged/train')
parser.add_argument("--encoder", type=str, default = 'efficientnet-b0')
parser.add_argument("--epoch_len", type=int, default=10)
parser.add_argument("--sup_batch_size", type=int, default=1)
parser.add_argument("--crop_size", type=int, default=512)
parser.add_argument("--workers", default=6, type=int)
parser.add_argument('--img_aug', type=str, default='d4')
parser.add_argument('--max_epochs', type=int, default=150)

parser.add_argument('--train_split_coef', type = float, default = 0.7)   
parser.add_argument('--encoder_weights', type = str, default = "imagenet")
parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
parser.add_argument('--training_mode', type = str, default = "encoder_decoder")
parser.add_argument('--strategy', type = str, default = "encoder_decoder")
parser.add_argument('--buffer_size', type = float, default = 0.1)

parser.add_argument('--len_seq', type = int, default=10)
parser.add_argument('--num_gpu', type = str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu

accuracy = Accuracy(num_classes=2)
training_mode = args.training_mode

def lambda_lr(epoch):    
    m = (epoch / args.max_epochs)*100
    if m < args.lr_milestones[0]:
        return 1
    elif m < args.lr_milestones[1]:
        return 1 + ((m - args.lr_milestones[0]) / (args.lr_milestones[1] - args.lr_milestones[0])) * (args.final_lr / args.initial_lr - 1)
    else:
        return args.final_lr/args.initial_lr
       
def train(model, train_dataloader, loss_fn, optimizer):
    loss_sum = 0.0
    acc_sum  = 0.0                    
    model.train()
    for i, batch in enumerate(train_dataloader):    
        image = batch['image'][:,:3,:,:]
        target = (batch['mask'])
        # image = norm_transforms(image).to(device)                        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()                         
        # Forward
        logits = model(image)            
        loss = loss_fn(logits, target)
        batch['preds'] = logits
        batch['image'] = image 
        loss.backward()
        optimizer.step()  # updating parameters                        
        acc_sum += accuracy(torch.transpose(logits,0,1).reshape(2, -1).t(), 
                            torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
        loss_sum += loss.item()                        
    

def validation(model, val_dataloader, loss_fn, scheduler):
    loss_sum = 0.0
    acc_sum = 0.0
    iou = 0.0
    precision = 0.0
    recall = 0.0  
    scheduler.step()
    model.eval()     
    for i, batch in enumerate(val_dataloader):
        image = batch['image']
        target = (batch['mask'])   
        # image = norm_transforms(image)                 
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
                         
    
def main():          
    # Exécution sur GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = np.random.randint(0, 1000)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True) 
    print(f"Seed: {seed}")
    
    # Récupérer la liste des sous-dossiers
    sous_dossiers = os.listdir(args.data_path)    
    random.seed(seed) 
    sequence_list = random.sample(sous_dossiers, args.len_seq)
    if not os.path.exists(args.sequence_path.format(seed)):
        os.makedirs(args.sequence_path.format(seed))  
    columns = ['run', 'ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time', 'method']
    
    train_imgs = []
    train_lbl = []
    
    test_imgs = []
    test_lbl = []
    
    step = 0
    
    idx = step-1 if step !=0 else step
    print(sequence_list)
    
    for domain in sequence_list:
        
        img = glob.glob(os.path.join(args.data_path, '{}/IMG_Z*_*.tif'.format(domain)))
        lbl = glob.glob(os.path.join(args.data_path, '{}/MSK_Z*_*.tif'.format(domain)))
        
        img, lbl = shuffle(np.array(img),np.array(lbl))
        
        img = img.tolist()
        lbl = lbl.tolist()
        
        train_imgs += img[:int(len(img)*args.train_split_coef)]
        train_lbl  += lbl[:int(len(lbl)*args.train_split_coef)]
        
        test_imgs += img[int(len(img)*args.train_split_coef):]
        test_lbl  += lbl[int(len(lbl)*args.train_split_coef):]
        
        domain_img = [item for item in train_imgs if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/IMG_Z*_*.tif'.format(domain)))]
        domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/MSK_Z*_*.tif'.format(domain)))]
        domain_img, domain_lbl = shuffle(np.array(domain_img),np.array(domain_lbl))
                
        domain_img = domain_img.tolist()
        domain_lbl = domain_lbl.tolist()
        
        domain_img_train = domain_img[:int(len(domain_img)*args.train_split_coef)]
        domain_lbl_train = domain_lbl[:int(len(domain_img)*args.train_split_coef)]
        
        domain_img_val = domain_img[int(len(domain_img)*args.train_split_coef):]
        domain_lbl_val = domain_lbl[int(len(domain_lbl)*args.train_split_coef):]
        
        
        if step !=0 and args.strategy == 'replay':
        
            coef_replay = args.buffer_size/3
            past_domain_img = []
            past_domain_lbl = []
            idx_past = 0 if step-5<0 else step-5
                
            for source_domain in sequence_list[idx_past:step]:
                a_domain_img = [item for item in train_imgs if fnmatch.fnmatch(item, os.path.join(args.data_path, 'images/{}*.tif'.format(source_domain)))]
                a_domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(args.data_path, 'gt/{}*.tif'.format(source_domain)))] 
                coef = int(len(a_domain_img)*coef_replay) if int(len(a_domain_img)*coef_replay)>0 else 1
                a_domain_img_train = a_domain_img[:coef]
                a_domain_lbl_train = a_domain_lbl[:coef]
                
                past_domain_img += a_domain_img_train
                past_domain_lbl += a_domain_lbl_train
            
            domain_img = [item for item in train_imgs if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/IMG_Z*_*.tif'.format(domain)))]
            domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/MSK_Z*_*.tif'.format(domain)))]
            
            domain_img = domain_img + past_domain_img
            domain_lbl = domain_lbl + past_domain_lbl
            
        elif args.strategy == 'baseline':
            domain_img = [item for item in train_imgs if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/IMG_Z*_*.tif'.format(domain)))]
            domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/MSK_Z*_*.tif'.format(domain)))]
            
        domain_img, domain_lbl = shuffle(np.array(domain_img),np.array(domain_lbl))
                
        domain_img = domain_img.tolist()
        domain_lbl = domain_lbl.tolist()
        
        domain_img_train = domain_img[:int(len(domain_img)*args.train_split_coef)]
        domain_lbl_train = domain_lbl[:int(len(domain_img)*args.train_split_coef)]
        
        domain_img_val = domain_img[int(len(domain_img)*args.train_split_coef):]
        domain_lbl_val = domain_lbl[int(len(domain_lbl)*args.train_split_coef):]                       
        
        # Train dataset
        train_datasets = []
        for img_path, lbl_path in zip( domain_img_train, domain_lbl_train):
            img = rasterio.open(img_path)
            train_datasets.append(FlairDs(image_path = img_path, label_path = lbl_path, fixed_crops = False,
                                tile=Window(col_off=0, row_off=0, width=img.shape[1], height=img.shape[0]),
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
        for img_path, lbl_path in zip(domain_img_val, domain_lbl_val):
             img = rasterio.open(img_path)
             val_datasets.append(FlairDs(image_path = img_path, label_path = lbl_path, fixed_crops = True,
                                tile=Window(col_off=0, row_off=0,  width=img.shape[1], height=img.shape[0]),
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

        print(step, training_mode)
        f=open(args.sequence_path.format(seed)+"/logfile{}.txt".format(seed),"a+")
        txt = " {} {} Step {} \n {}\n".format(args.strategy,training_mode,step, sequence_list)
        # f.write(txt)
        f.close()
        model_path= os.path.join(args.sequence_path.format(seed), '{}{}_step{}{}'.format(args.strategy,seed, step, training_mode))
        # UNet SS task
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights if args.train_with_void else None ,
            in_channels=args.in_channels,
            classes=args.num_classes,
            decoder_use_batchnorm=True)
        model.to(device)       
        
                
        if step == 1 :                     
            saved_model_path = os.path.join(args.sequence_path.format(seed), '{}{}_step{}{}'.format(args.strategy,seed, int(idx), training_mode))
        else :     
            saved_model_path = os.path.join(args.sequence_path.format(seed), '{}{}_step{}{}'.format(args.strategy,seed, int(idx), training_mode))
        model.load_state_dict(torch.load(saved_model_path))
            
        if training_mode == "decoder":                    
            for param in model.encoder.parameters():
                param.requires_grad = False 
            optimizer = SGD(
                model.decoder.parameters(),
                lr=args.initial_lr,
                momentum=0.9)
        elif training_mode == "encoder_decoder":                    
            optimizer = SGD(
                model.parameters(),
                lr=args.initial_lr,
                momentum=0.9)
            
        loss_fn = torch.nn.BCEWithLogitsLoss()     
        early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=model_path)
        
        train_visu = (model_path+"_res")
        train_writer = SummaryWriter(train_visu)    
        
        seg_img_visu = SegmentationImagesVisualisation(writer = train_writer,freq = 10)
        scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
        accuracy = Accuracy(num_classes=2)
        start_epoch = 0
        
        for epoch in range(start_epoch, args.max_epochs):
            loss_sum = 0.0
            acc_sum  = 0.0                    
            time_ep = time.time()
            
            train(model, train_dataloader, loss_fn, optimizer)
            
            seg_img_visu.display_batch(train_writer, batch, 5,epoch,prefix='train')                    
            train_loss = {'loss': loss_sum / len(train_dataloader)}
            train_acc = {'acc': acc_sum/len(train_dataloader)}
            train_writer.add_scalar('Loss/train', train_loss['loss'], epoch+1)
            train_writer.add_scalar('Acc/train', train_acc['acc'], epoch+1)
            
            validation(model, val_dataloader, loss_fn, scheduler)
            
            seg_img_visu.display_batch(train_writer, batch,5, epoch,prefix = 'val') 
            
            val_iou = {'iou':iou/len(val_dataloader)}
            val_precision = {'prec':precision/len(val_dataloader)}
            val_recall = {'recall':recall/len(val_dataloader)}
            val_loss = {'loss': loss_sum / len(val_dataloader)} 
            val_acc = {'acc': acc_sum/ len(val_dataloader)}
            
            early_stopping(val_loss['loss'], model)
            if early_stopping.early_stop:
                    print("Early Stopping")
                    break   
            time_ep = time.time() - time_ep
            train_writer.add_scalar('Loss/val', val_loss['loss'], epoch+1)
            train_writer.add_scalar('Acc/val', val_acc['acc'], epoch+1)
            train_writer.add_figure('Confusion matrix', plot_confusion_matrix(cm.cpu(), class_names = ['0','1']), epoch+1)
            train_writer.add_scalar('IoU/val', val_iou['iou'], epoch+1)
            train_writer.add_scalar('Prec/val', val_precision['prec'], epoch+1)
            train_writer.add_scalar('Recall/val', val_recall['recall'], epoch+1)
            values = [seed, epoch+1, train_loss['loss'], val_loss['loss'],train_acc['acc'],val_acc['acc'], time_ep, training_mode]
            table = tabulate.tabulate([values], columns, tablefmt='simple',
                                      floatfmt='8.4f')
            print(table)
            
        train_writer.add_graph(model, image)
        train_writer.flush()
        train_writer.close()
        f=open(args.sequence_path.format(seed)+"/logfile{}.txt".format(seed),"a+")
        txt = "DONE"
        f.write(txt)
        f.close()
    step += 1
            
if __name__ == "__main__":
    main()    
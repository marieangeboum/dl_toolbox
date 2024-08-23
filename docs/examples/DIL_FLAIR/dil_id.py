import os
import glob
import time
import tabulate
import fnmatch
import random
import segmentation_models_pytorch as smp

from rasterio.windows import Window
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import Accuracy

import dl_toolbox.inference as dl_inf
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.networks import MultiHeadUnet
from dl_toolbox.callbacks import *
from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_datasets import *
from dl_toolbox.torch_datasets.utils import *

parser = ArgumentParser()
parser.add_argument("--train_with_void", action='store_true')
parser.add_argument("--eval_with_void", action='store_true')
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--seed", type=int)
parser.add_argument("--in_channels", type=int, default=5)
parser.add_argument("--num_classes", type=int, default = 1)    
parser.add_argument("--initial_lr", type=float, default = 0.001)
parser.add_argument("--final_lr", type=float, default = 0.0005)
parser.add_argument("--lr_milestones", nargs=2, type=float, default=(20,80))
parser.add_argument("--data_path", type=str, default ='/scratchf/CHALLENGE_IGN/FLAIR_1/train')
parser.add_argument("--encoder", type=str, default = 'efficientnet-b2')
parser.add_argument("--epoch_len", type=int, default=10000)
parser.add_argument("--sup_batch_size", type=int, default=8)
parser.add_argument("--crop_size", type=int, default=256)
parser.add_argument("--workers", default=6, type=int)
parser.add_argument('--img_aug', type=str, default='d4')
parser.add_argument('--max_epochs', type=int, default=120)
parser.add_argument('--tile_width', type = int, default = 256)
parser.add_argument('--tile_height', type = int, default = 256)
parser.add_argument('--train_split_coef', type = float, default = 0.7)   
parser.add_argument('--encoder_weights', type = str, default = "imagenet")
parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
parser.add_argument('--strategy', type = str, default = "replay")
parser.add_argument('--pretrain', type = str, default = "in")
parser.add_argument('--buffer_size', type = float, default = 0.2)
parser.add_argument('--len_seq', type = int, default=10)
parser.add_argument('--num_gpu', type = str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lambda_lr(epoch):    
    m = (epoch / args.max_epochs)*100
    if m < args.lr_milestones[0]:
        return 1
    elif m < args.lr_milestones[1]:
        return 1 + ((m - args.lr_milestones[0]) / (args.lr_milestones[1] - args.lr_milestones[0])) * (args.final_lr / args.initial_lr - 1)
    else:
        return args.final_lr/args.initial_lr
       
    
def main():          
    try:
      seed =  1571
      print(f"Seed: {seed}")
      torch.manual_seed(seed)
      torch.cuda.manual_seed(seed)
      torch.autograd.set_detect_anomaly(True) 
      
      columns = ['strategy','run', 'step','ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time', 'method']   
      
      list_of_tuples = [(item, sequence_list.index(item)) for item in sequence_list]
            
      if not os.path.exists(args.sequence_path.format(seed)):
          os.makedirs(args.sequence_path.format(seed))  
      
      f=open(args.sequence_path.format(seed)+"/logfile_id{}.txt".format(seed),"a+")
      txt = "Sequence{} : \n".format(seed)
      f.write(txt)
      f.write("Encoder : {}\n".format(args.encoder))        
      f.close()
      
      train_imgs = []    
      test_imgs = []
      
      step = 0
      idx = step-1 if step !=0 else step
            
      for domain in sequence_list:
          if step == 0 and args.strategy == 'replay' :
              step += 1
              continue
          
          img = glob.glob(os.path.join(args.data_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
          random.shuffle(img)
          train_imgs += img[:int(len(img)*args.train_split_coef)]
          # train_lbl  += lbl[:int(len(lbl)*args.train_split_coef)]
          test_imgs += img[int(len(img)*args.train_split_coef):]
          # test_lbl  += lbl[int(len(lbl)*args.train_split_coef):]       
          
          if args.strategy == 'baseline':
              domain_img = [item for item in train_imgs if  fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
              # domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/Z*_*/msk/MSK_*.tif'.format(domain)))]
              
          if step !=0 and args.strategy == 'replay':
          
              coef_replay = args.buffer_size/5
              past_domain_img = []
              # past_domain_lbl = []
              idx_past = 0 if step-5<0 else step-5
                  
              for source_domain in sequence_list[idx_past:step]:
                  a_domain_img = [item for item in train_imgs if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
                  # a_domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/Z*_*/msk/MSK_*.tif'.format(domain)))] 
                  coef = int(len(a_domain_img)*coef_replay) if int(len(a_domain_img)*coef_replay)>0 else 1
                  a_domain_img_train = a_domain_img[:coef]
                  # a_domain_lbl_train = a_domain_lbl[:coef]
                  past_domain_img += a_domain_img_train
                  # past_domain_lbl += a_domain_lbl_train
              
              domain_img = [item for item in train_imgs if  fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
              # domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/Z*_*/msk/MSK_*.tif'.format(domain)))]
              domain_img = domain_img + past_domain_img
              # domain_lbl = domain_lbl + past_domain_lbl
              
          
          random.shuffle(domain_img)            
          domain_img_train = domain_img[:int(len(domain_img)*args.train_split_coef)]        
          domain_img_val = domain_img[int(len(domain_img)*args.train_split_coef):]       
          # Train dataset
          train_datasets = []
          for img_path in domain_img_train : 
              img_path_strings = img_path.split('/')
              domain_pattern = img_path_strings[-4]
              img_pattern = img_path_strings[-1].split('_')[-1].strip('.tif')
              lbl_path = glob.glob(os.path.join(args.data_path, '{}/Z*_*/msk/MSK_{}.tif'.format(domain_pattern, img_pattern)))[0]
              train_datasets.append(FlairDs(image_path = img_path, label_path = lbl_path, fixed_crops = False,
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
              collate_fn = CustomCollate(batch_aug = args.img_aug),
              num_workers=args.workers)  
          
          # Validation dataset
          val_datasets = []
          for img_path in domain_img_val : 
              img_path_strings = img_path.split('/')
              domain_pattern = img_path_strings[-4]
              img_pattern = img_path_strings[-1].split('_')[-1].strip('.tif')
              lbl_path = glob.glob(os.path.join(args.data_path, '{}/Z*_*/msk/MSK_{}.tif'.format(domain_pattern, img_pattern)))[0]
              # for img_path, lbl_path in zip(domain_img_val, domain_lbl_val):             
              val_datasets.append(FlairDs(image_path = img_path, label_path = lbl_path, fixed_crops = True,
                                  tile=Window(col_off=0, row_off=0,  width=args.tile_width, height=args.tile_height),
                                  crop_size=args.crop_size,
                                  crop_step=args.crop_size,
                                  img_aug=args.img_aug))
               
          valset =  ConcatDataset(val_datasets)
          val_dataloader = DataLoader(
              dataset=valset,
              shuffle=False,
              batch_size=args.sup_batch_size,
              collate_fn = CustomCollate(batch_aug = args.img_aug),
              num_workers=args.workers)  
  
        
if __name__ == "__main__":
    main()   
    
    
# continuer redac articles
# faire code pour replay avec id 
# lancer code pour replay avec id
# lancer code sur plusieurs run pour replay simple 
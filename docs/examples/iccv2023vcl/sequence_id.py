import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import glob
import numpy as np
import time
import tabulate
import torch
import fnmatch
import random
import rasterio
import subprocess

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
from dl_toolbox.networks import MultiHeadUnet
#from dl_toolbox.multiheadsmp.Unet import  MultiHeadsmp.Unet
from dl_toolbox.callbacks import SegmentationImagesVisualisation, CustomSwa, ConfMatLogger
from dl_toolbox.callbacks import plot_confusion_matrix, compute_conf_mat, EarlyStopping
from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_datasets import *
from dl_toolbox.torch_datasets.utils import *



parser = ArgumentParser()

parser.add_argument("--train_with_void", action='store_true')
parser.add_argument("--eval_with_void", action='store_true')
parser.add_argument("--pretrained", action='store_true')
#parser.add_argument("--output_dir", type=str, default="./outputs")
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
      sequence_list =  ['D072_2019', 'D031_2019', 'D023_2020', 'D009_2019', 'D014_2020', 'D007_2020', 'D004_2021', 'D080_2021', 'D055_2018', 'D035_2020']
      list_of_tuples = [(item, sequence_list.index(item)) for item in sequence_list]
      print(list_of_tuples)
      
      if args.pretrain == 'in':
          args.sequence_path ='in_sequence_{}/'          
      if args.pretrain == 'flair1':
         args.sequence_path ='flair1_sequence_{}/'          
      if args.pretrain == 'inria':
          args.sequence_path ='inria_sequence_{}/'          
      if args.pretrain == 'no':
          args.sequence_path ='none_sequence_{}/' 
          
      if not os.path.exists(args.sequence_path.format(seed)):
          os.makedirs(args.sequence_path.format(seed))  
      
      print(sequence_list)
      f=open(args.sequence_path.format(seed)+"/logfile_id{}.txt".format(seed),"a+")
      txt = "Sequence{} : \n".format(seed)
      f.write(txt)
      f.write("Encoder : {}\n".format(args.encoder))        
      f.close()
      # # Récupérer la liste des sous-dossiers
      # sous_dossiers = os.listdir(args.data_path)    
      # random.seed(seed) 
      # sequence_list = random.sample(sous_dossiers, args.len_seq)
      
      columns = ['strategy','run', 'step','ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time', 'method']   
      # norm_transforms = transforms.Compose([transforms.Normalize(InriaAllDs.stats['mean'], InriaAllDs.stats['std'])])
      
      train_imgs = []
      train_lbl = []
      
      test_imgs = []
      test_lbl = []
      
      # step = 0
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
          
          for training_mode in ("model", "decoder"):
              print(step, training_mode)
              f=open(args.sequence_path.format(seed)+"/progress_logfile_id{}.txt".format(seed),"a+")
              txt = " {} {} Step {} \n {}\n".format(args.strategy,training_mode,step, sequence_list)
              f.write(txt)
              f.close()
              model_path= os.path.join(args.sequence_path.format(seed), 'id_{}_{}_step{}{}'.format(args.strategy,seed, step, training_mode)) 
              if step == 0 :
                  idx = step
                  if training_mode == 'decoder': 
                      # step += 1
                      continue
                  model_path= os.path.join(args.sequence_path.format(seed), 'id_{}_step{}{}'.format(seed, step, training_mode))
   
              # if args.pretrain == 'flair1': 
                  
              #     # smp.Unet SS task
              #     model = smp.Unet(
              #         encoder_name=args.encoder,
              #         encoder_weights= None ,
              #         in_channels=args.in_channels,
              #         classes=args.num_classes,
              #         decoder_use_batchnorm=True)
              #     model.load_state_dict(torch.load('pretraining_flair1_5Ds_3'))
              # #     model.to(device)  
              # if args.pretrain =='inria':
              #     # smp.Unet SS task
              #     model = smp.Unet(
              #         encoder_name=args.encoder,
              #         encoder_weights= None ,
              #         in_channels=args.in_channels,
              #         classes=args.num_classes,
              #         decoder_use_batchnorm=True)
              #     model.load_state_dict(torch.load('all_seed_3939'))
              #     model.to(device)
              # if args.pretrain == 'in':
              #     # smp.Unet SS task
              #     model = smp.Unet(
              #         encoder_name=args.encoder,
              #         encoder_weights= args.encoder_weights if args.train_with_void else None ,
              #         in_channels=args.in_channels,
              #         classes=args.num_classes,
              #         decoder_use_batchnorm=True)
              #     model.to(device)
                  
              # if args.pretrain == 'no':
              #   # smp.Unet SS task
              model = smp.Unet(
                    encoder_name=args.encoder,
                    encoder_weights=  None,
                    in_channels=args.in_channels,
                    classes=args.num_classes,
                    decoder_use_batchnorm=True)
              model.to(device) 
              multi_head_model = MultiHeadUnet(model, args.len_seq, head_idx = list_of_tuples[step][1])
                
              multi_head_model.to(device)      
              if step == 1 :  
                  idx = step-1                   
                  saved_model_path = os.path.join(args.sequence_path.format(seed), 'id_{}_step{}{}'.format(seed, int(idx), "model"))
                  multi_head_model.load_state_dict(torch.load(saved_model_path))
                  # multi_head_model = MultiHeadUnet(model, args.len_seq, head_idx = list_of_tuples[step][1])
                  # multi_head_model.load_state_dict(torch.load(saved_model_path))
                  multi_head_model.to(device) 
                  
              elif step!=0 and step!= 1:
                  idx = step-1              
                  saved_model_path = os.path.join(args.sequence_path.format(seed), 'id_{}_{}_step{}{}'.format(args.strategy,seed, int(idx), training_mode))
                  multi_head_model.load_state_dict(torch.load(saved_model_path))               
                  # multi_head_model = MultiHeadUnet(model, args.len_seq, head_idx = list_of_tuples[step][1])
                  
                  multi_head_model.to(device) 
                  
              
              # Set all parameters except one in a specific segmentation head to requires_grad = False
              
              for i, head in enumerate(multi_head_model.segmentation_heads):
                  if i <= list_of_tuples[step][1]:  # Specify the target head index
                      for param in head.parameters():
                          param.requires_grad = True
                  else:
                      for param in head.parameters():
                          param.requires_grad = False
  
              if training_mode == "decoder":            
                  
                  for param in multi_head_model.encoder.parameters():
                      param.requires_grad = False 
                  optimizer = SGD(
                      multi_head_model.decoder.parameters(),
                      lr=args.initial_lr,
                      momentum=0.9)
              elif training_mode == "model":                    
                  optimizer = SGD(
                      multi_head_model.parameters(),
                      lr=args.initial_lr,
                      momentum=0.9)
                  
              loss_fn = torch.nn.BCEWithLogitsLoss()     
              early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=model_path)
              
              train_visu = (model_path+"_res")
              train_writer = SummaryWriter(train_visu)    
              
              seg_img_visu = SegmentationImagesVisualisation(writer = train_writer,freq = 10)
              scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
              accuracy = Accuracy(num_classes=2).cuda()
              start_epoch = 0
              
              for epoch in range(start_epoch, args.max_epochs):
                  loss_sum = 0.0
                  acc_sum  = 0.0                    
                  time_ep = time.time()
                  
                  for i, batch in enumerate(train_dataloader):                     
                      image = batch['image'][:,:args.in_channels,:,:].to(device)
                      target = (batch['mask']).to(device)                    
                      image = (image/255.).to(device)   
                      dom_id = dict(list_of_tuples)[batch['id']] 
                      # Clear gradients w.r.t. parameters
                      optimizer.zero_grad()                         
                      logits = multi_head_model(image)
                      
                      batch['preds'] = logits
                      batch['image'] = image 
                      loss = loss_fn(logits, target)
                      loss.backward()
                      optimizer.step()  # updating parameters                        
                      acc_sum += accuracy(torch.transpose(logits,0,1).reshape(2, -1).t(), 
                                          torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
                      loss_sum += loss.item()                  
                  
                  seg_img_visu.display_batch(train_writer, batch, 5,epoch,prefix='train')      
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
                  multi_head_model.eval()
                       
                  for i, batch in enumerate(val_dataloader):
                      image = (batch['image'][:,:args.in_channels,:,:]).to(device)
                      target = (batch['mask']).to(device)  
                      image = (image/255.).to(device)                
                      output = multi_head_model(image)  
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
                  seg_img_visu.display_batch(train_writer, batch,5, epoch,prefix = 'val')
                                  
                  val_iou = {'iou':iou/len(val_dataloader)}
                  val_precision = {'prec':precision/len(val_dataloader)}
                  val_recall = {'recall':recall/len(val_dataloader)}
                  val_loss = {'loss': loss_sum / len(val_dataloader)} 
                  val_acc = {'acc': acc_sum/ len(val_dataloader)}
                  
                  early_stopping(val_loss['loss'],multi_head_model)
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
                  values = [args.strategy,seed, step,epoch+1, train_loss['loss'], val_loss['loss'],train_acc['acc'],val_acc['acc'], time_ep, training_mode]
                  table = tabulate.tabulate([values], columns, tablefmt='simple',
                                            floatfmt='8.4f')
                  print(table)
                  f=open(args.sequence_path.format(seed)+"/progress_logfile_id{}.txt".format(seed),"a+")
                  f.write(txt)
                  f.write(table)
                  f.close()
                  
              train_writer.add_graph(multi_head_model, image)
              train_writer.flush()
              train_writer.close()
          step += 1
      f=open(args.sequence_path.format(seed)+"/test_logfile_id{}.txt".format(seed),"a+")
      txt = "Sequence{} : \n".format(seed)
      f.write(txt)
      f.write("Encoder : {}\n".format(args.encoder))
      for item_x in test_imgs :
          f.write(item_x + '\n')             
      f.close()
    except Exception as e:
      # Handle the exception
      error_message = str(e)
  
      # Write the error message to a text file
      with open('error.txt', 'w') as file:
          print(error_message)
          file.write(error_message)
        
if __name__ == "__main__":
    main()   
    
    
# continuer redac articles
# faire code pour replay avec id 
# lancer code pour replay avec id
# lancer code sur plusieurs run pour replay simple 
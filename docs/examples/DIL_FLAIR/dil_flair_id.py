import os

import json
import glob
import time as t 
import  numpy as np
import tabulate
import fnmatch
import random
import logging
import traceback

import segmentation_models_pytorch as smp
from datetime import datetime, time
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
from argparse import ArgumentParser
from datetime import datetime
from collections import defaultdict

# Configure the logging settings
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')
parser = ArgumentParser()
# Préparation du jeu de données
parser.add_argument("--metadata_file", type=str, default = '/scratchf/CHALLENGE_IGN/FLAIR_1/flair-one_metadata.json')
parser.add_argument("--target_camera", type=str, default ='UCE')
parser.add_argument("--target_year", type=str, default ='2020')
parser.add_argument("--target_zone", type=str, default ='UU')
parser.add_argument("--time_slot", nargs=2, type=float, default=(8,11))
# Args pour l'entrainement
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
parser.add_argument("--sup_batch_size", type=int, default=16)
parser.add_argument("--crop_size", type=int, default=256)
parser.add_argument("--workers", default=6, type=int)
parser.add_argument('--img_aug', type=str, default='d4')
parser.add_argument('--max_epochs', type=int, default=120)
parser.add_argument('--tile_width', type = int, default = 512)
parser.add_argument('--tile_height', type = int, default = 512)
parser.add_argument('--train_split_coef', type = float, default = 0.7)   
parser.add_argument('--encoder_weights', type = str, default = "imagenet")
parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
parser.add_argument('--strategy', type = str, default = "replay")
parser.add_argument('--buffer_size', type = float, default = 0.2)
parser.add_argument('--num_gpu', type = str, default = "1")
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
    
def is_time_in_timeslot(check_time, start_time, end_time):
    check_datetime = datetime.combine(datetime.today(), check_time)
    start_datetime = datetime.combine(datetime.today(), start_time)
    end_datetime = datetime.combine(datetime.today(), end_time)
    return start_datetime <= check_datetime <= end_datetime    
   
def main():  
        
    target_zone = args.target_zone
    target_camera = args.target_camera
    target_year = args.target_year
    time_slot = args.time_slot
    file_path = args.metadata_file
    time_format = '%Hh%M'
    start_time_slot = time(args.time_slot[0],0)
    end_time_slot = time(args.time_slot[1],0)
    
    try:
        with open(file_path, "r") as file:
            metadata = json.load(file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON data from '{file_path}'.")
    
    filtered_data = {key: value for key, value in metadata.items() if value['zone'].split('_')[1] == target_zone 
                     and value['camera'].split('-')[0] == target_camera and value['date'].split('-')[0] == target_year and  is_time_in_timeslot(datetime.strptime(value['time'], time_format).time(), start_time_slot, end_time_slot) == True}
    imgs_name_dataset = list(filtered_data.keys())
    
    # Create a new dictionary to store sets of unique values for each key
    unique_values_filtered_data = {}
    for sub_dict in filtered_data.values():
        for key, value in sub_dict.items():
            if key == 'zone':
                value = value.split('_')[1]
            if key not in unique_values_filtered_data:
                unique_values_filtered_data[key] = set()
            unique_values_filtered_data[key].add(value)
    
    #Get domains list
    domains_list = list(unique_values_filtered_data['domain'])
    # Create a dictionary to store the count of occurrences for each unique value
    occurrence_counts = defaultdict(lambda: defaultdict(int))
    for sub_dict in filtered_data.values():
        for key, value in sub_dict.items():
            # print(key)
            if key == 'zone':
                value = value.split('_')[1]          
            occurrence_counts[key][value] += 1    
    paths_dataset ={}
    paths = []
    for domain in domains_list :
        # Récuperer les chemins de chaque image et ensuite les selectionner par domaines
        paths.extend(glob.glob(os.path.join(args.data_path, '{}/Z*_{}/img/IMG_*.tif'.format(domain, target_zone))))
        paths_dataset[domain] = [item for item in paths if item.split('/')[-1].strip('.tif') in imgs_name_dataset and item.split('/')[4] == domain]
    # paths_dataset = {key: del value for key, value in paths_dataset.items() if len(value) == 0}
    # Definition du train/test data     
    try:
        seed = args.seed if args.strategy == 'replay' else np.random.randint(0, 5000)
        print(f"Seed: {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.autograd.set_detect_anomaly(True) 
        # Volume de données passées à conserver
        coef_replay = args.buffer_size/5
        columns = ['strategy','run', 'step','ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time']   
        sequence_list = domains_list
        list_of_tuples = [(item, sequence_list.index(item)) for item in sequence_list]
        dict_of_tuples = dict(list_of_tuples)      
        if not os.path.exists(args.sequence_path.format(seed)):
            os.makedirs(args.sequence_path.format(seed))  
        
        f=open(args.sequence_path.format(seed)+"/logfile_id{}.txt".format(seed),"a+")
        txt = "Sequence {} : \n".format(seed)
        f.write(txt)
        f.write("Encoder : {}\n".format(args.encoder))  
        print(target_zone, target_camera, target_year, time_slot, start_time_slot, end_time_slot, sep = '\n', file = f)
        print(domains_list, file = f)
        f.close()
        
        train_imgs = {}    
        test_imgs = {}
        # Train/ Test for each domain in sequence    
        for domain in sequence_list:  
            imgs_list = paths_dataset[domain]
            print(domain)      
            shuffled_imgs_list = random.sample(imgs_list, len(imgs_list))
            train_imgs[domain]=shuffled_imgs_list[:int(len(imgs_list)*args.train_split_coef)]
            test_imgs[domain]= shuffled_imgs_list[int(len(imgs_list)*args.train_split_coef):]
        
        # Write the dictionary to the file in JSON format
        with open(args.sequence_path.format(seed)+"/test_logfile_id_{}.json".format(seed), "w") as file:
            json.dump(test_imgs, file, indent=4)
        with open(args.sequence_path.format(seed)+"/train_logfile_id_{}.json".format(seed), "w") as file:
            json.dump(train_imgs, file, indent=4)
        
        # Définition du buffer 
        past_domain_img = {}                 
        # Définition de l'ensemble de données
       
        for domain, step in list_of_tuples :
            idx = step-1 if step !=0 else step
            idx_past = 0 if step-5<=0 else step-5
            print(step, domain)          
            
            train_domain = train_imgs[domain]     
            # Stockage des données dans un buffer
            past_imgs = train_imgs[domain]     
            shuffled_past = random.sample(past_imgs, len(past_imgs))
            past_domain_img[domain] = shuffled_past[:int(len(past_imgs)*coef_replay)] 
            
            # Select 5 previous elements for each element in the list
            past_domains_label = sequence_list[max(0, step - 5):step]
            replay_buffer = [item for key in  past_domains_label for item in past_domain_img.get(key, [])]
    
            if args.strategy == 'replay' : 
               train_domain += replay_buffer 
            domain_img_train = train_domain[:int(len(train_domain)*args.train_split_coef)]        
            domain_img_val = train_domain[int(len(train_domain)*args.train_split_coef):] 
                    
            # Train dataset
            train_datasets = []
            for img_path in domain_img_train: 
                img_path_strings = img_path.split('/')
                domain_pattern = img_path_strings[-4]
                img_pattern = img_path_strings[-1].split('_')[-1].strip('.tif')
                lbl_path = glob.glob(os.path.join(args.data_path, '{}/Z*_{}/msk/MSK_{}.tif'.format(domain_pattern, target_zone,img_pattern)))[0]
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
                lbl_path = glob.glob(os.path.join(args.data_path, '{}/Z*_{}/msk/MSK_{}.tif'.format(domain_pattern,target_zone, img_pattern)))[0]
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
            
            f=open(args.sequence_path.format(seed)+"/progress_logfile_id{}.txt".format(seed),"a+")
            print(args.strategy,step, sequence_list, sep = '\n', file = f)
            f.close()
            # Définition de modèles
            model_path= os.path.join(args.sequence_path.format(seed), 'id_{}_{}_step{}'.format(args.strategy,seed, step)) 
            if step == 0 :
                idx = step
                model_path= os.path.join(args.sequence_path.format(seed), 'id_{}_step{}'.format(seed, step))
    
            model = smp.Unet(
                  encoder_name=args.encoder,
                  encoder_weights=  None,
                  in_channels=args.in_channels,
                  classes=args.num_classes,
                  decoder_use_batchnorm=True)
            model.to(device) 
            multi_head_model = MultiHeadUnet(model, num_heads = len(sequence_list), domain_ids_list=[0])
              
            multi_head_model.to(device)
            if step == 1 :  
                idx = step-1                   
                saved_model_path = os.path.join(args.sequence_path.format(seed), 'id_{}_step{}'.format(seed, int(idx)))
                multi_head_model.load_state_dict(torch.load(saved_model_path))
                multi_head_model.to(device) 
                
            elif step!=0 and step!= 1:
                idx = step-1              
                saved_model_path = os.path.join(args.sequence_path.format(seed), 'id_{}_{}_step{}'.format(args.strategy,seed, int(idx)))
                multi_head_model.load_state_dict(torch.load(saved_model_path))               
                multi_head_model.to(device)
                
            for i, head in enumerate(multi_head_model.segmentation_heads):
                if i <= list_of_tuples[step][1]:  # Specify the target head index
                    for param in head.parameters():
                        param.requires_grad = True
                        
                else:
                    for param in head.parameters():
                        param.requires_grad = False
           
            early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=model_path)
            train_writer_comment = (model_path+"_visualisation")
            train_writer = SummaryWriter(train_writer_comment)   
            image_segmentation_visualisation = SegmentationImagesVisualisation(writer = train_writer,freq = 10)
            
            optimizer = SGD(
                multi_head_model,
                lr=args.initial_lr,
                momentum=0.9)
            loss_fn = torch.nn.BCEWithLogitsLoss() 
            scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
            accuracy = Accuracy(num_classes=2).cuda()
            start_epoch = 0
            for epoch in range(start_epoch, args.max_epochs):
                loss_sum = 0.0
                acc_sum  = 0.0                    
                time_ep = t.time()
                for i, batch in enumerate(train_dataloader):                     
                    image = (batch['image']/255.).to(device)
                    target = (batch['mask']).to(device)                    
                    domain_ids = [list_of_tuples[dict_of_tuples[element]][1] for element in batch['id']]
                    optimizer.zero_grad()                         
                    logits, shared_logits = multi_head_model(image, domain_ids_list= domain_ids)
                    loss = loss_fn(logits, target)
                    batch['preds'] = logits
                    batch['image'] = image 
                    loss.backward()
                    optimizer.step()  # updating parameters                        
                    acc_sum += accuracy(torch.transpose(logits,0,1).reshape(2, -1).t(), 
                                        torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
                    loss_sum += loss.item()                  
                
                image_segmentation_visualisation.display_batch(train_writer, batch, 10,epoch,prefix='train')      
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
                    image = (batch['image']/255.).to(device)
                    target = (batch['mask']).to(device)  
                    domain_ids = [list_of_tuples[dict_of_tuples[element]][1] for element in batch['id']] 
                    output, shared_output = multi_head_model(image,domain_ids_list= domain_ids)  
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
                image_segmentation_visualisation.display_batch(train_writer, batch,5, epoch,prefix = 'val')
                val_iou = {'iou':iou/len(val_dataloader)}
                val_precision = {'prec':precision/len(val_dataloader)}
                val_recall = {'recall':recall/len(val_dataloader)}
                val_loss = {'loss': loss_sum / len(val_dataloader)} 
                val_acc = {'acc': acc_sum/ len(val_dataloader)}
                
                early_stopping(val_loss['loss'],multi_head_model)
                if early_stopping.early_stop:
                        print("Early Stopping")
                        break
                time_ep = t.time() - time_ep
                train_writer.add_scalar('Loss/val', val_loss['loss'], epoch+1)
                train_writer.add_scalar('Acc/val', val_acc['acc'], epoch+1)
                train_writer.add_figure('Confusion matrix', plot_confusion_matrix(cm.cpu(), class_names = ['0','1']), epoch+1)
                train_writer.add_scalar('IoU/val', val_iou['iou'], epoch+1)
                train_writer.add_scalar('Prec/val', val_precision['prec'], epoch+1)
                train_writer.add_scalar('Recall/val', val_recall['recall'], epoch+1)
                values = [args.strategy,seed, step,epoch+1, train_loss['loss'], val_loss['loss'],train_acc['acc'],val_acc['acc'], time_ep]
                table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
                print(table)
                f=open(args.sequence_path.format(seed)+"/progress_logfile_id{}.txt".format(seed),"a+")
                f.write(txt)
                f.write(table)
                f.close()
            train_writer.add_graph(multi_head_model, image)
            train_writer.flush()
            train_writer.close()
    except Exception as e:
        # Handle the exception    
        error_message = f"An error occurred: {str(e)}"
        error_trace = traceback.format_exc()  # Get the formatted stack trace
        logging.error("%s\n%s", error_message, error_trace)
        # Write the error message to a text file        
        print(error_message, error_trace)
if __name__ == "__main__":
    main()   
              

            
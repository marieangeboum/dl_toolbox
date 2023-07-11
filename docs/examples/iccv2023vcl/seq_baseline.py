import os
import glob
import numpy as np
import time
import tabulate
import torch
import fnmatch
import random

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


def main():   

    # parser for argument easier to launch .py file and inintalizing arg. correctly
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_classes", type=int, default = 1)
    parser.add_argument("--train_with_void", action='store_true')
    parser.add_argument("--eval_with_void", action='store_true')
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--encoder", type=str, default = 'mit_b0')
    parser.add_argument("--initial_lr", type=float, default = 0.01)
    parser.add_argument("--final_lr", type=float, default = 0.005)
    parser.add_argument("--lr_milestones", nargs=2, type=float, default=(20,80))
    parser.add_argument("--data_path", type=str, default ='/scratchf/flair_merged/train')
    
    # parser.add_argument("--data_path", type=str, default = '/scratchf/DATASETS/INRIA/AerialImageDataset/train')
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='d4')
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--train_split_coef', type = float, default = 0.7)
    parser.add_argument('--tile_width', type = int, default = 2560)
    parser.add_argument('--tile_height', type = int, default = 2560)
    # parser.add_argument('--model_path', type = str, default ="sequence_{}/{}_{}_seed_{}")
    parser.add_argument('--encoder_weights', type = str, default = "imagenet")
    parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
    # parser.add_argument('-n', '--sequence_list', nargs='+', default=[])
    parser.add_argument('--len_seq', type = int, default=10)
    args = parser.parse_args()
        
    # Exécution sur GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    seed = np.random.randint(0, 1000)
    print(f"Seed: {seed}")
    # Récupérer la liste des sous-dossiers
    sous_dossiers = os.listdir(args.data_path)
    # np.random.seed(seed)     
    random.seed(seed) 
    sequence_list = random.sample(sous_dossiers,15)
    print(sequence_list)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True) # stops training if something is wrong
    # Learning rate 
    def lambda_lr(epoch):    
        m = (epoch / args.max_epochs)*100
        if m < args.lr_milestones[0]:
            return 1
        elif m < args.lr_milestones[1]:
            return 1 + ((m - args.lr_milestones[0]) / (args.lr_milestones[1] - args.lr_milestones[0])) * (args.final_lr / args.initial_lr - 1)
        else:
            return args.final_lr/args.initial_lr
           
        
    columns = ['run', 'ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time', 'method']
    norm_transforms = transforms.Compose([transforms.Normalize(InriaAllDs.stats['mean'], InriaAllDs.stats['std'])])
   
    # A domain
    train_imgs = []
    train_lbl = []
    
    test_imgs = []
    test_lbl = []
        
    if not os.path.exists(args.sequence_path.format(seed)):
        os.makedirs(args.sequence_path.format(seed))
    print(args.sequence_path.format(seed))   
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
    
     
        f=open(args.sequence_path.format(seed)+"/logfile{}.txt".format(seed),"a+")
        txt = "Sequence{} : \n".format(seed)
        
        
        
        f.write("Encoder : {}".format(args.encoder))
        for item_x, item_y in zip(test_imgs, test_lbl) :
            f.write(item_x + '\n') 
            f.write(item_y + '\n\n')
        f.close()  
       
        step = 0   
        print(sequence_list)
        
        a_domain_img = [item for item in train_imgs if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/IMG_Z*_*.tif'.format(domain)))]
        a_domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(args.data_path, '{}/MSK_Z*_*.tif'.format(domain)))]
        a_domain_img, a_domain_lbl = shuffle(np.array(a_domain_img),np.array(a_domain_lbl))
                
        a_domain_img = a_domain_img.tolist()
        a_domain_lbl = a_domain_lbl.tolist()
        
        a_domain_img_train = a_domain_img[:int(len(a_domain_img)*args.train_split_coef)]
        a_domain_lbl_train = a_domain_lbl[:int(len(a_domain_img)*args.train_split_coef)]
        
        a_domain_img_val = a_domain_img[int(len(a_domain_img)*args.train_split_coef):]
        a_domain_lbl_val = a_domain_lbl[int(len(a_domain_lbl)*args.train_split_coef):]
        
        # Train dataset
        train_datasets = []
        for img_path, lbl_path in zip( a_domain_img_train, a_domain_lbl_train):
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
            collate_fn = CustomCollate(batch_aug = 'no'),
            num_workers=args.workers)        
        
        # Validation dataset
        val_datasets = []
        for img_path, lbl_path in zip(a_domain_img_val, a_domain_lbl_val):
             val_datasets.append(FlairDs(image_path = img_path, label_path = lbl_path, fixed_crops = True,
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
        print(step)

        for training_mode in ("encoder_decoder", "decoder"):
            
            print(step, training_mode)
            f=open(args.sequence_path.format(seed)+"/logfile{}.txt".format(seed),"a+")
            txt = "\nBaseline {} Step {} \n {}\n".format(training_mode,step, sequence_list)
            f.write(txt)
            f.close()
            model_path= os.path.join(args.sequence_path.format(seed), 'seq{}_step{}{}'.format(seed, step, training_mode))
            # if step == 1 :                     
            #     saved_model_path = os.path.join(args.sequence_path.format(seed), 'seq{}_step{}'.format(seed, int(step-1)))
            # else : 
            if step == 0 : 
                idx = step
                if training_mode == 'decoder' : 
                    continue
                # UNet SS task
                model = smp.Unet(

                    encoder_name=args.encoder,
                    encoder_weights=args.encoder_weights if args.train_with_void else None ,
                    in_channels=args.in_channels,
                    classes=1,
                    decoder_use_batchnorm=True)
                model.to(device)
            else : 
                idx = step-1
                saved_model_path = os.path.join(args.sequence_path.format(seed), 'seq{}_step{}{}'.format(seed, int(idx), training_mode))
                # UNet SS task
                model = smp.Unet(
                    encoder_name=args.encoder,
                    encoder_weights=args.encoder_weights if args.train_with_void else None ,
                    in_channels=args.in_channels,
                    classes=1,
                    decoder_use_batchnorm=True)
                model.to(device)
                model.load_state_dict(torch.load(saved_model_path))
            
            if training_mode == "decoder":                    
                for param in model.encoder.parameters():
                    param.requires_grad = False 
                # initializing the optimizer
                optimizer = SGD(
                    model.decoder.parameters(),
                    lr=args.initial_lr,
                    momentum=0.9)
            elif training_mode == "encoder_decoder":                    
                # initializing the optimizer
                optimizer = SGD(
                    model.parameters(),
                    lr=args.initial_lr,
                    momentum=0.9)
            # Initialize Loss
            # print(optimizer.state_dict()['param_groups'][0]['lr'])  
            loss_fn = torch.nn.BCEWithLogitsLoss()     
            early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=model_path)
            train_visu = (model_path+"_res")
            train_writer = SummaryWriter(train_visu)    
            seg_img_visu = SegmentationImagesVisualisation(writer = train_writer,freq = 10)
            scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
            accuracy = Accuracy(task = 'binary',num_classes=2).cuda()
            start_epoch = 0
            for epoch in range(start_epoch, args.max_epochs):
                loss_sum = 0.0
                acc_sum  = 0.0                    
                time_ep = time.time()
                model.train()
                for i, batch in enumerate(train_dataloader):    
                    image = batch['image'][:,:3,:,:].to(device)
                    print(image.shape)
                    target = (batch['mask']/255.).to(device)
                    # image = norm_transforms(image).to(device)                        
                    # Clear gradients w.r.t. parameters
                    optimizer.zero_grad()                         
                    # Forward
                    logits = model(image)            
                    loss = loss_fn(logits, target)
                    batch['preds'] = logits
                    batch['image'] = image                        
                    # Getting gradients w.r.t. parametersCUDA_VISIBLE_DEVICES=0 python3 seq_baseline.py --sequence_list  chicago kitsap austin tyrol-w vienna & CUDA_VISIBLE_DEVICES=0 python3 seq_baseline.py --sequence_list  chicago kitsap austin tyrol-w vienna & CUDA_VISIBLE_DEVICES=1 python3 seq_baseline.py --sequence_list  chicago kitsap austin tyrol-w vienna & CUDA_VISIBLE_DEVICES=1 python3 seq_baseline.py --sequence_list  chicago kitsap austin tyrol-w vienna & CUDA_VISIBLE_DEVICES=2 python3 seq_baseline.py --sequence_list  chicago kitsap austin tyrol-w vienna
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
                            
                seg_img_visu.display_batch(train_writer, batch,20, epoch,prefix = 'val') 
                
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
                values = [seed, epoch + 1, train_loss['loss'], val_loss['loss'],train_acc['acc'],val_acc['acc'], time_ep, training_mode]
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
        step +=1
    

if __name__ == "__main__":

    main()

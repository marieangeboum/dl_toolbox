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
from dl_toolbox.torch_datasets import InriaDs, InriaAustinDs, InriaViennaDs, InriaAllDs,  InriaChicagoDs, InriaKitsapDs, InriaTyrolDs
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
    parser.add_argument('--model_path', type = str, default ="unet_{}_{}")
    #parser.add_argument('--saved_model_path',type = str, default = ")
    args = parser.parse_args()
    
    # Exécution sur GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    torch.autograd.set_detect_anomaly(True) # stops training if something is wrong
    if args.img_aug == 'radio':
        pixel_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=1.0),
        A.ChannelShuffle(p=0.5),
        ])
    
    train_visu = (args.model_path+"_result_viz").format(args.data_source, args.img_aug)   
    train_writer = SummaryWriter(train_visu)    
    seg_img_visu = SegmentationImagesVisualisation(writer = train_writer,freq = 10)
    
    inference_visu = (args.model_path+"_inference_{}".format(args.data_target))
    inference_writer = SummaryWriter(inference_visu)
    inference_seg_img_visu = SegmentationImagesVisualisation(writer = inference_writer,freq = 1)
    columns = ['ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time']
      
    early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=args.model_path.format(args.data_source, args.img_aug))
    
    if args.data_source == 'austin':
        norm_transforms = transforms.Compose([transforms.Normalize(InriaAustinDs.stats['mean'], InriaAustinDs.stats['std'])])
        if args.data_target == 'vienna':
            norm_source_target = transforms.Compose([transforms.Normalize(InriaAustinViennaDs.stats['mean'], InriaAustinViennaDs.stats['std'])])
        if args.data_target == 'chicago':
            norm_source_target = transforms.Compose([transforms.Normalize(InriaAustinChicagoDs.stats['mean'], InriaAustinChicagoDs.stats['std'])])
        if args.data_target == 'tyrol-w':
            norm_source_target = transforms.Compose([transforms.Normalize(InriaAustinTyrolDs.stats['mean'], InriaAustinTyrolDs.stats['std'])])
            
    # Tiles list extraction
    data_both_image_paths_list = []
    data_both_label_paths_list = []
    for data in [args.data_source, args.data_target]:        
        data_both_image_paths_list.append(glob.glob(os.path.join(args.data_path, 'images/{}*.tif'.format(data))))
        data_both_label_paths_list.append(glob.glob(os.path.join(args.data_path, 'gt/{}*.tif'.format(data))))
    
    data_target_image_paths_list = glob.glob(os.path.join(args.data_path, 'images/{}*.tif'.format(args.data_target)))
    data_target_label_paths_list = glob.glob(os.path.join(args.data_path, 'gt/{}*.tif'.format(args.data_target)))
    
    # Train/Validation split
    train_img_paths_list = data_source_image_paths_list[:int(len(data_source_image_paths_list)*args.split_coef)]
    train_lbl_paths_list = data_source_label_paths_list[:int(len(data_source_label_paths_list)*args.split_coef)]
    val_img_paths_list = data_source_image_paths_list[int(len(data_source_image_paths_list)*args.split_coef):]
    val_lbl_paths_list = data_source_label_paths_list[int(len(data_source_label_paths_list)*args.split_coef):]
    
    test_img_paths_list = data_target_image_paths_list[:10]
    test_lbl_paths_list = data_target_label_paths_list[:10]        
    # Train dataset
    train_datasets = []
    for image_path, label_path in zip(train_img_paths_list, train_lbl_paths_list):
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
    for image_path, label_path in zip(val_img_paths_list, val_lbl_paths_list):
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
    
    # Test dataset
    test_datasets = []
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

    # UNet SS task
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=args.in_channels,
        classes=1,
        decoder_use_batchnorm=True)
    model.to(device)
    
    # Initialize Loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Initializing SGD optimizer
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
    
    #accuracy = Accuracy(task = 'binary',num_classes=2).cuda()
    accuracy = Accuracy(num_classes=2).cuda()
    
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
        seg_img_visu.display_batch(train_writer, batch, 5,epoch,prefix='train')
        train_loss = {'loss': loss_sum / len(train_dataloader)}
        train_acc = {'acc': acc_sum/len(train_dataloader)}
        writer.add_scalar('Loss/train', train_loss['loss'], epoch+1)
        writer.add_scalar('Acc/train', train_acc['acc'], epoch+1)
               
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
                    
        seg_img_visu.display_batch(train_writer, batch,5, epoch,prefix = 'val') 
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
    inference_model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_use_batchnorm=True)
    
    inference_model.to(device)
    inference_model.load_state_dict(torch.load(args.model_path))
    inference_model.eval()
    acc_sum = 0.0
    iou     = 0.0
    precision = 0.0
    recall = 0.0
    
    #perf_df = pd.DataFrame(columns = ['crop_name','F1',  'Recall' ,'Precision ' , 'IoU', 'macroRecall', 'macroPrecision', 'crop_path'])
    crop_name, Recall, Precision, IoU = [],[],[],[]
    # accuracy = Accuracy(task = 'binary',num_classes=2).cuda()
    accuracy = Accuracy(num_classes=2).cuda()
    with torch.no_grad(): 
        for i, batch in enumerate(test_dataloader):
            
            image = batch['image'].to(device)
            target = (batch['mask']/255.).to(device)
            image = norm_transforms(image).to(device) 
            img = batch['orig_image'][0].cpu().numpy().transpose(1, 2, 0)/255
            batch['image'] = image
            output = model(image)  
            batch['preds'] = output.cpu()
            
            cm = compute_conf_mat(
                    torch.tensor(target).flatten().cpu(),
                    torch.tensor((torch.sigmoid(output)>0.5).cpu().long().flatten().cpu()), 2)
            metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy()) 
            
            iou += metrics_per_class_df.IoU[1]
            precision += metrics_per_class_df.Precision[1]
            recall += metrics_per_class_df.Recall[1]
            acc_sum += accuracy(torch.transpose(output,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
            
            img = batch['orig_image'][0].cpu().numpy().transpose(1, 2, 0)/255
            if os.path.isfile('./{}/orig_image_{}_num_{}.jpeg'.format(viz_file,args.data_target, i)):
                print('./{}/orig_image_{}_num_{}.jpeg'.format(viz_file,args.data_target, i),'already exists', sep = ' ')
            
            crop_name.append('orig_image_{}_num_{}'.format(args.data_target, i))
            
            Recall.append(metrics_per_class_df.Recall[1])
            Precision.append(metrics_per_class_df.Precision[1])
            IoU.append(metrics_per_class_df.IoU[1])
            # matplotlib.image.imsave('./{}/orig_image_{}_num_{}.jpeg'.format(viz_file,args.data_target, i),img)
            inference_seg_img_visu.display_batch(inference_writer,batch, 1,i,prefix='orig_image_{}_num_{}'.format(args.data_target, i))   
        
        test_acc = acc_sum/ len(test_dataloader)   
        test_iou = iou/len(test_dataloader)
        test_precision = precision/len(test_dataloader)
        test_recall = recall/len(test_dataloader)
        
    perf_df = pd.DataFrame({'crop_name' : crop_name, 'Recall' : Recall,'Precision ': Precision , 'IoU':IoU})
    with pd.ExcelWriter('inference_metrics.xlsx', mode="a",  if_sheet_exists='overlay') as writer_xlsx:  
        perf_df.to_excel(writer_xlsx, sheet_name=viz_file, index=False)
    with open( viz_file+'.txt', mode = 'w')  as f : 
        f.write("test_acc : "+str(np.float64(test_acc.item()))+'\n\n')
        f.write("test_iou :"+str(np.float64(test_iou))+'\n\n')
        f.write("test_precision "+str(np.float64(test_precision))+'\n\n')
        f.write("test_recall "+str(np.float64(test_recall))+'\n\n')
    
if __name__ == "__main__":

    main()

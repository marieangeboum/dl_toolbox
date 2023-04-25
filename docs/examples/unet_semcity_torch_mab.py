import os
import glob

import time
import tabulate

import torch
import matplotlib.pyplot as plt
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
from dl_toolbox.torch_datasets import InriaDs, InriaAustinDs, InriaAllDs
from dl_toolbox.torch_datasets.utils import *

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature


#def main():

   # parser for argument easier to launch .py file and inintalizing arg. correctly
parser = ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./outputs")
parser.add_argument("--num_classes", type=int)
parser.add_argument("--train_with_void", action='store_true')
parser.add_argument("--eval_with_void", action='store_true')
parser.add_argument("--in_channels", type=int, default = 3)
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--encoder", type=str)
parser.add_argument("--initial_lr", type=float)
parser.add_argument("--final_lr", type=float)
parser.add_argument("--lr_milestones", nargs=2, type=float)
parser.add_argument("--data_path", type=str)
parser.add_argument("--epoch_len", type=int, default=5000)
parser.add_argument("--sup_batch_size", type=int, default=16)
parser.add_argument("--crop_size", type=int, default=256)
parser.add_argument("--workers", default=6, type=int)
parser.add_argument('--img_aug', type=str, default='no')
parser.add_argument('--max_epochs', type=int, default=500)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--townA', type = str, default = 'austin')
args = parser.parse_args()

# execution sur GPU si  ce dernier est dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# temporaire  à changer pour mettre le nom de ville en args
townB_image_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/{}*.tif'.format(args.townA)))
townB_label_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/{}*.tif'.format(args.townA)))

# temporaire  à changer pour mettre le nom de ville en args
townA_image_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'images/{}*.tif'.format(args.townA)))
townA_label_paths_list = glob.glob(os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/{}*.tif'.format(args.townA)))

coef = 0.7
train_img_paths_list = townA_image_paths_list[:int(len(townA_image_paths_list)*coef)]
train_lbl_paths_list = townA_label_paths_list[:int(len(townA_label_paths_list)*coef)]

val_img_paths_list = townA_image_paths_list[int(len(townA_image_paths_list)*coef):]
val_lbl_paths_list = townA_label_paths_list[int(len(townA_label_paths_list)*coef):]

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
for image_path, label_path in zip(train_img_paths_list, train_lbl_paths_list):
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
val_dataloader = DataLoader(
    dataset=valset,
    shuffle=False,
    batch_size=args.sup_batch_size,
    collate_fn = CustomCollate(batch_aug = 'no'),
    num_workers=args.workers)

# Unet model for SS task
model = smp.Unet(
    encoder_name='efficientnet-b0',
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_use_batchnorm=True)
model.to(device)

# unet_model = UNet(3, n_classes=1)
# unet_model.to(device)
# A changer pour le masking ("reduction=none")
# initializing loss function
loss_fn = torch.nn.BCEWithLogitsLoss()
print(loss_fn)
grey_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3)])
norm_transforms = transforms.Compose([transforms.Normalize(InriaAustinDs.stats['mean'], InriaAustinDs.stats['std'])])
# initializing the optimizer
optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9)

# definition of learning rate
def lambda_lr(epoch):
    m = epoch / 100
    if m < 10:
        return 1
    elif m < 100:
        return 1 + ((m - 10) / (100 - 10)) * (0.0005/0.01 - 1)
    else:
        return 0.0005/0.01



scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
#viz = SegmentationImagesVisualisation(writer)
start_epoch = 0
columns = ['ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time']
writer = SummaryWriter("INRIA--smp_unet_{}--epoch_len-{}-sup_batch_size-{}".format(args.townA, args.epoch_len,args.sup_batch_size))
#writer = SummaryWriter("test")
viz = SegmentationImagesVisualisation(writer = writer,freq = 10)
early_stopping = EarlyStopping(patience=10, verbose=True,  delta=0.001,path='smp_unet_{}_austin_norm.pt'.format(args.townA))

for epoch in range(start_epoch, args.max_epochs):
    

    time_ep = time.time()
    loss_sum = 0.0
    acc_sum = 0.0
    
    model.train()
    #unet_model.train()
    for i, batch in enumerate(train_dataloader):
        
        image = batch['image'].to(device)
        target = (batch['mask']/255.).to(device)
                
        image = norm_transforms(image).to(device)
        #image = grey_transforms(image).to(device)
                               
        # clear gradients wrt parameters
        optimizer.zero_grad()
        # mask processing to do here ???
        ###########################
        # forward phase to get outputs
        
        logits = model(image)        
        #logits = unet_model(image)]
        
        batch['preds'] = logits
        batch['image'] = image
        # compute loss
        loss = loss_fn(logits, target) 
      
        # getting gradients wrt parameters
        loss.backward()
        # updating parameters
        optimizer.step()   
        accuracy = Accuracy(num_classes=2).cuda()
        acc_sum += accuracy(torch.transpose(logits,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
        loss_sum += loss.item()
        print("LOSS: ",loss.item())
        
   
    viz.display_batch(writer, batch, 10,epoch,prefix='train')
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
    #unet_model.eval()
    
    for i, batch in enumerate(val_dataloader):
        
        image = batch['image'].to(device)
        target = (batch['mask']/255.).to(device)
        
        image = norm_transforms(image).to(device)
        #image = grey_transforms(image).to(device)
        output = model(image)
               
        #output = unet_model(image)
        batch['preds'] = output
        batch['image'] = image
        
        cm = compute_conf_mat(
            torch.tensor(target).flatten().cpu(),
            torch.tensor((torch.sigmoid(output)>0.5).cpu().long().flatten().cpu()), 2)
        
        metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy()) 
        
        iou += metrics_per_class_df.IoU[1]
        precision += metrics_per_class_df.Precision[1]
        recall += metrics_per_class_df.Recall[1]
        
        loss = loss_fn(output, target) # computing loss <> predicted output and labels
        loss_sum += loss.item()
        
        accuracy = Accuracy(num_classes=2).cuda()
        acc_sum += accuracy(torch.transpose(output,0,1).reshape(2, -1).t(), torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
        
    viz.display_batch(writer, batch,10, epoch,prefix = 'val')
    
    val_iou = iou/len(val_dataloader)
    val_precision = precision/len(val_dataloader)
    val_recall = recall/len(val_dataloader)
    
    val_loss = {'loss': loss_sum / len(val_dataloader)} #
    val_acc = {'acc': acc_sum/ len(val_dataloader)}
    early_stopping(val_loss['loss'], model)
    
    if early_stopping.early_stop:
            print("Early stopping")
            break
        
    writer.add_scalar('Loss/val', val_loss['loss'], epoch+1)
    writer.add_scalar('Acc/val', val_acc['acc'], epoch+1)
    writer.add_figure('Confusion matrix', plot_confusion_matrix(cm.cpu(), class_names = ['0','1']), epoch+1)
    writer.add_scalar('IoU/val', val_iou, epoch+1)
    writer.add_scalar('Prec/val', val_precision, epoch+1)
    writer.add_scalar('Recall/val', val_recall, epoch+1)
    
    #metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy())
    #metrics_per_class_df.rename(index=dict([(i, l) for i, l in enumerate(dataset.labels.keys())]),inplace=True)
 
    time_ep = time.time() - time_ep
    values = [epoch + 1, train_loss['loss'], val_loss['loss'],train_acc['acc'],val_acc['acc'], time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple',
                              floatfmt='8.4f')
    print(table)
writer.add_graph(model, image)
writer.flush()
writer.close()
#torch.save(model.state_dict(),'smp_unet_{}_norm_panchro.pt'.format(args.townA))

#if __name__ == "__main__":

#    main()

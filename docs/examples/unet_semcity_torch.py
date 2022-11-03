from argparse import ArgumentParser
import torch
from rasterio.windows import Window
from torch.utils.tensorboard import SummaryWriter
from dl_toolbox.callbacks import SegmentationImagesVisualisation, CustomSwa, ConfMatLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging

import os
import glob
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_datasets import InriaDs
import segmentation_models_pytorch as smp
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from dl_toolbox.torch_collate import CustomCollate
import time
import tabulate



#writer = SummaryWriter()
def main():
    writer = SummaryWriter("INRIA -- pytorch unet model trained on Vienna -- fromscratch -- 5")

    # parser for argument easier to launch .py file and inintalizing arg. correctly
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--train_with_void", action='store_true')
    parser.add_argument("--eval_with_void", action='store_true')
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--initial_lr", type=float)
    parser.add_argument("--final_lr", type=float)
    parser.add_argument("--lr_milestones", nargs=2, type=float)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='no')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    #parser.add_argument('--townB', type = str)
    args = parser.parse_args()
    # execution sur GPU si  ce dernier est dispo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.autograd.set_detect_anomaly(True) # stops training if something is wrong
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # temporaire  à changer pour mettre le nom de ville en args
    townA_image_paths_list = glob.glob(os.path.join(args.data_path, 'images//austin*.tif'))
    townA_label_paths_list = glob.glob(os.path.join(args.data_path, 'gt//austin*.tif'))
    print(len(townA_image_paths_list), len(townA_label_paths_list))

    coef = 0.8
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
    #print(len(train_dataloader.dataset))
    val_dataloader = DataLoader(
        dataset=valset,
        shuffle=False,
        batch_size=args.sup_batch_size,
        collate_fn = CustomCollate(batch_aug = 'no'),
        num_workers=args.workers)
    #print(len(val_dataloader.dataset))

    # Unet model for SS task
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights='imagenet' if args.pretrained else None,
        in_channels=args.in_channels,
        classes=args.num_classes if args.train_with_void else args.num_classes - 1,
        decoder_use_batchnorm=True)
    model.to(device)

    # A changer pour le masking ("reduction=none")
    # initializing loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(loss_fn)

    # initializing the optimizer
    #optimizer = Adam(model.parameters(), lr=0.01)
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

    start_epoch = 0
    columns = ['ep', 'train_loss', 'val_loss','train_acc','val_acc', 'time']
    print("Loop over train_dataset X times")
    for epoch in range(start_epoch, args.max_epochs):

        time_ep = time.time()
        loss_sum = 0.0
        num_examples_train = 0.0
        num_correct_train = 0.0
        model.train()

        for i, batch in enumerate(train_dataloader):

            image = batch['image'].to(device)
            target = batch['mask'].to(device)/255.
            # clear gradients wrt parameters
            optimizer.zero_grad()
            # mask processing to do here ???
            #######################
            # forward to get outputs
            logits = model(image)
            #print("LOGITS",logits)
            # calculate loss
            #######################
            loss = loss_fn(logits, target)
            #print("LOSS", loss)
            correct_train = torch.eq(torch.softmax(logits, 1).argmax(1),target).view(-1)
            num_correct_train += torch.sum(correct_train).item()
            num_examples_train += correct_train.shape[0]
            # getting gradients wrt parameters
            loss.backward()
            # updating parameters
            optimizer.step()

            loss_sum += loss.item()
            
            print("LOSS SUM: ",loss_sum)

        print(len(train_dataloader))
        train_loss = {'loss': loss_sum / len(train_dataloader)}
        train_acc = {'acc': num_correct_train/num_examples_train}
        

        loss_sum = 0.0
        num_examples_val = 0.0
        num_correct_val = 0.0
        
        scheduler.step()

        model.eval()
        print("Loop over val dataset X times ")
        for i, batch in enumerate(val_dataloader):

            image = batch['image'].to(device)
            target = batch['mask'].to(device)/255.

            output = model(image) # use this output in display_batch func
            #print(output)

            loss = loss_fn(output, target) # computing loss <> predicted output and labels
            print(loss)
            
            correct_val  = torch.eq(torch.softmax(output, 1).argmax(1),target).view(-1)
            num_correct_val += torch.sum(correct_val).item()
            num_examples_val += correct_val.shape[0]
            
            loss_sum += loss.item()
            
            

        
        val_loss = {'loss': loss_sum / len(val_dataloader)} 
        val_acc = {'acc': num_correct_val/ num_examples_val}
        

        time_ep = time.time() - time_ep
        print(time_ep)
        values = [epoch + 1, train_loss['loss'], val_loss['loss'],train_acc['acc'],val_acc['acc'], time_ep]
        table = tabulate.tabulate([values], columns, tablefmt='simple',
                                  floatfmt='8.4f')
        
        writer.add_scalar('Loss/train', train_loss['loss'], epoch+1)
        writer.add_scalar('Loss/val', val_loss['loss'], epoch+1)
        writer.add_scalar('Acc/train', train_acc['acc'], epoch+1)
        writer.add_scalar('Acc/val', val_acc['acc'], epoch+1)
        print(table)
    writer.flush()
    writer.close()
    torch.save(model.state_dict(),'smp_unet_vienna_fromscratch5.pt')
# do not forget to save model once evrthing is good

if __name__ == "__main__":

    main()

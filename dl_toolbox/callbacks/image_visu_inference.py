# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning.utilities import rank_zero_warn
import numpy as np
from matplotlib import cm


class SegmentationImagesVisualisationInf(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    NB_COL: int = 2

    def __init__(self, writer,freq, *args, **kwargs):

        super().__init__(*args, **kwargs)
        #self.visu_fn = visu_fn # conversion of labels into --> ??
        self.freq = freq
        self.writer = writer

#    def on_train_batch_end(
#            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#    ) -> None:

#        if trainer.current_epoch % self.freq == 0 and batch_idx == 0:
#
#            self.display_batch(
#                trainer,
#                outputs['batch'],
#                prefix='Train'
#        )

    def display_batch(self, writer, batch,freq, epoch, prefix):
        if epoch % self.freq == 0:

            img = batch['image'].cpu()
            orig_img = batch['orig_image'].cpu()
            target = batch['mask'].cpu()/255
            preds = batch['preds'].cpu() # output de unet_semcity_torch
            #print(type(preds))
            
            #preds_rgb = self.visu_fn(preds).transpose((0,3,1,2))
            #np_preds_rgb = torch.from_numpy(preds.detach().numpy()).float() # transform numpy array into torch tensor
            #np_preds = 255 - torch.from_numpy(preds.detach().numpy().astype(np.uint8)).float()
            # à faire avec sigmoid
            np_preds =torch.from_numpy(torch.sigmoid(preds).detach().numpy()).float()
            
            magma = cm.get_cmap('viridis')
            
            np_preds_cmap =torch.from_numpy(magma(np_preds)).squeeze(1).permute(0,3,1,2)[:,:3,:,:]
            
            
            np_preds_int = torch.round(np_preds).cpu()
            if batch['mask'] is not None:
                labels = batch['mask'].cpu()
                #labels_rgb = self.visu_fn(labels).transpose((0,3,1,2))
                np_labels = torch.from_numpy(labels.detach().numpy()).float()
                #np_labels_rgb = torch.from_numpy(labels_rgb).float()
                
           
                 
            TP = (target.int() & np_preds_int.int()) 
            FP = ((target.int() - np_preds_int) == -1)
            FN = ((target.int()==1 )& (np_preds_int.int() == 0))
            overlay = torch.cat([FN,np_preds_int,FP], dim =1).float().clip(0,1)
            heat_map_pred = torch.mul(np_preds, overlay)
            # Number of grids to log depends on the batch size
            quotient, remainder = divmod(img.shape[0], self.NB_COL)
            nb_grids = quotient + int(remainder > 0)
    
            for idx in range(nb_grids):
    
                start = self.NB_COL * idx
                if start + self.NB_COL <= img.shape[0]:
                    end = start + self.NB_COL
                else:
                    end = start + remainder
                    
                img_grid = torchvision.utils.make_grid(img[start:end, :, :, :], padding=10, normalize=True)
                orig_img_grid = torchvision.utils.make_grid(orig_img[start:end, :, :, :], padding=10, normalize=True)
                out_grid = torchvision.utils.make_grid(np_preds_cmap[start:end, :, :, :], padding=10, normalize=False)
                #out_grid_ceil = torchvision.utils.make_grid(overlay_ceil[start:end, :, :, :], padding=10, normalize=True)
                
                error_grid = torchvision.utils.make_grid(overlay[start:end, :, :, :], padding=10, normalize=True)
                heat_maps_grid = torchvision.utils.make_grid(heat_map_pred[start:end, :, :, :], padding=10, normalize=True)
                grids = [orig_img_grid,img_grid, error_grid, out_grid]
    
                # if batch['mask'] is not None:
                #     mask_grid = torchvision.utils.make_grid(np_labels[start:end, :, :, :], padding=10, normalize=True)
                #     #print(np_labels.shape())
                #     grids.append(mask_grid)
                    
                    
                final_grid = torch.cat(grids, dim=1)
                
                #trainer.logger.experiment.add_image(f'Images/{prefix}_batch_art_{idx}', final_grid, global_step=trainer.global_step)
                writer.add_image(f'Images/{prefix}_batch_art_{idx}', final_grid, epoch+1)

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the validation batch ends."""

        if trainer.current_epoch % self.freq == 0 and batch_idx == 0:
            self.display_batch(trainer, outputs['batch'], prefix='Val')

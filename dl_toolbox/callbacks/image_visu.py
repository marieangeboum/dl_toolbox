# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning.utilities import rank_zero_warn
import numpy as np


class SegmentationImagesVisualisation(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    NB_COL: int = 2

    def __init__(self, writer, *args, **kwargs):

        super().__init__(*args, **kwargs)
        #self.visu_fn = visu_fn # conversion of labels into --> ??
        #self.freq = freq
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

    def display_batch(self, writer, batch, prefix):

        img = batch['image'].cpu()
        orig_img = batch['orig_image'].cpu()
        preds = batch['preds'].cpu() # output de unet_semcity_torch
        #print(type(preds))
        
        #preds_rgb = self.visu_fn(preds).transpose((0,3,1,2))
        #np_preds_rgb = torch.from_numpy(preds.detach().numpy()).float() # transform numpy array into torch tensor
        np_preds= torch.from_numpy(preds.detach().numpy().astype(np.uint8)).float()
        
        if batch['mask'] is not None:
            labels = batch['mask'].cpu()
            #labels_rgb = self.visu_fn(labels).transpose((0,3,1,2))
            np_labels = torch.from_numpy(labels.detach().numpy()).float()
            #np_labels_rgb = torch.from_numpy(labels_rgb).float()

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
            out_grid = torchvision.utils.make_grid(np_preds[start:end, :, :, :], padding=10, normalize=True)
            grids = [orig_img_grid, img_grid, out_grid]

            if batch['mask'] is not None:
                mask_grid = torchvision.utils.make_grid(np_labels[start:end, :, :, :], padding=10, normalize=True)
                #print(np_labels.shape())
                grids.append(mask_grid)
                
                
            final_grid = torch.cat(grids, dim=1)

            #trainer.logger.experiment.add_image(f'Images/{prefix}_batch_art_{idx}', final_grid, global_step=trainer.global_step)
            writer.add_image(f'Images/{prefix}_batch_art_{idx}', final_grid, 0)

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the validation batch ends."""

        if trainer.current_epoch % self.freq == 0 and batch_idx == 0:
            self.display_batch(trainer, outputs['batch'], prefix='Val')

from torch.utils.data._utils.collate import default_collate
import torch
from dl_toolbox.torch_datasets.utils import *

class CustomCollate(): # call in inria_ds batch_aug = no

    def __init__(self, batch_aug):
        self.batch_aug = get_transforms(batch_aug)

    def __call__(self, batch, *args, **kwargs):
        # batch list de touts les dict des images extracted
        windows = [elem['window'] for elem in batch if 'window' in elem.keys()] # list of  all les window
        keys_to_collate = ['image', 'orig_image','id', 'mask', 'orig_mask'] # liste des autres element  tensor qui sont envoyé au default collate
        to_collate = [{k: v for k, v in elem.items() if (k in keys_to_collate) and (v is not None)} for elem in batch]
        batch = default_collate(to_collate)
        if 'mask' not in batch.keys():
            batch['mask'] = None
        batch['image'], batch['mask'] = self.batch_aug(batch['image'], batch['mask'])
        batch['window'] = windows

        return batch # 1 dict with all elemnt return parcous of dataloader

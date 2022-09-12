from argparse import ArgumentParser
import segmentation_models_pytorch as smp
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
import torch
import torchmetrics.functional as torchmetrics
from dl_toolbox.losses import DiceLoss
from copy import deepcopy
import torch.nn.functional as F

from dl_toolbox.lightning_modules.utils import *

class BaseModule(pl.LightningModule):

    # Validation step common to all modules if possible

    def __init__(self,
                 num_classes,
                 weights,
                 ignore_index,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_classes = num_classes
        self.weights = list(weights)
        self.ignore_index = ignore_index

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--weights", type=int, nargs="+")
        parser.add_argument("--ignore_index", type=int)

        return parser
    
    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)
        scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.lr_milestones,
            gamma=0.2
        )

        return [self.optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        logits = self.forward(inputs)
        probas = logits.softmax(dim=1)
        confidences, preds = torch.max(probas, dim=1)

        batch['probas'] = probas.detach()
        batch['confs'] = confidences.detach()
        batch['preds'] = preds.detach()
        batch['logits'] = logits.detach()

        stat_scores = torchmetrics.stat_scores(
            preds,
            labels,
            ignore_index=self.ignore_index if self.ignore_index >= 0 else None,
            mdmc_reduce='global',
            reduce='macro',
            num_classes=self.num_classes
        )
        
        loss1 = self.loss1(logits, labels)
        #loss2 = self.loss2(logits, labels)
        loss2=0
        loss = loss1 + loss2
        self.log('Val_CE', loss1)
        self.log('Val_Dice', loss2)
        self.log('Val_loss', loss)

        return {'batch': batch,
                'stat_scores': stat_scores.detach(),
                }

    def validation_epoch_end(self, outs):
        
        stat_scores = [out['stat_scores'] for out in outs]

        class_stat_scores = torch.sum(torch.stack(stat_scores), dim=0)
        f1_sum = 0
        tp_sum = 0
        supp_sum = 0
        nc = 0
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        for i in range(num_classes):
            if i != self.ignore_index:
                tp, fp, tn, fn, supp = class_stat_scores[i, :]
                if supp > 0:
                    nc += 1
                    f1 = tp / (tp + 0.5 * (fp + fn))
                    self.log(f'Val_f1_{i}', f1)
                    f1_sum += f1
                    tp_sum += tp
                    supp_sum += supp
        
        self.log('Val_acc', tp_sum / supp_sum)
        self.log('Val_f1', f1_sum / nc) 

    def on_train_epoch_end(self):
        for param_group in self.optimizer.param_groups:
            self.log(f'learning_rate', param_group['lr'])
            break

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:39:30 2022

@author: maboum

"""

# See https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py 

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve and IoU doesn't increase after a given patience."""

    def __init__(self, patience=7, verbose=False, delta_loss=0.03, delta_iou=0.01, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved and IoU increased.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss and IoU improvement. 
                            Default: False
            delta_loss (float): Minimum change in the monitored validation loss to qualify as an improvement.
                            Default: 0.03
            delta_iou (float): Minimum change in the monitored IoU to qualify as an improvement.
                            Default: 0.01
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss_score = None
        self.best_iou_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_iou_max = 0.0
        self.delta_loss = delta_loss
        self.delta_iou = delta_iou
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, val_iou, model):
        loss_score = -val_loss
        iou_score = val_iou

        # Check if both validation loss improved and IoU increased
        if (self.best_loss_score is None or loss_score < self.best_loss_score + self.delta_loss) and \
           (self.best_iou_score is None or iou_score > self.best_iou_score + self.delta_iou):
            self.best_loss_score = loss_score
            self.best_iou_score = iou_score
            self.save_checkpoint(val_loss, val_iou, model)
            self.counter = 0  # Reset counter when both conditions are met
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, val_iou, model):
        '''Saves model when both validation loss decreases and IoU increases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}) and IoU increased ({self.val_iou_max:.6f} --> {val_iou:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.val_iou_max = val_iou

    def save_state(self):
        """Save the state of EarlyStopping."""
        return {
            'counter': self.counter,
            'best_loss_score': self.best_loss_score,
            'best_iou_score': self.best_iou_score,
            'val_loss_min': self.val_loss_min,
            'val_iou_max': self.val_iou_max,
            'early_stop': self.early_stop
        }

    def load_state(self, state):
        """Load the state of EarlyStopping."""
        self.counter = state['counter']
        self.best_loss_score = state['best_loss_score']
        self.best_iou_score = state['best_iou_score']
        self.val_loss_min = state['val_loss_min']
        self.val_iou_max = state['val_iou_max']
        self.early_stop = state['early_stop']

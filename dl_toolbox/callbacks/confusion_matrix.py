import pytorch_lightning as pl
import torch
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Necessary for imshow to run on machines with no graphical interface.
plt.switch_backend("agg")

# Taken from https://www.tensorflow.org/tensorboard/image_summaries
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    tick_marks = np.arange(len(class_names))
    
    sum_col = torch.sum(cm,dim=1, keepdim=True)
    sum_lin = torch.sum(cm,dim=0, keepdim=True)
    cm_recall = torch.nan_to_num(cm/sum_col, nan=0., posinf=0., neginf=0.).numpy()
    cm_precision = torch.nan_to_num(cm/sum_lin, nan=0., posinf=0., neginf=0.).numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
    
    ax1.imshow(cm_recall, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Recall matrix")
    ax1.set_xticks(tick_marks, class_names)
    ax1.set_yticks(tick_marks, class_names)
    # Compute the labels from the normalized confusion matrix.
    # labels = np.around(cm.astype("float"), decimals=2)
    labels = np.around(cm_recall.astype('float') * 100).astype('int')
    # Use white text if squares are dark; otherwise black.
    threshold = cm_recall.max() / 2.0
    for i, j in itertools.product(range(cm_recall.shape[0]), range(cm_recall.shape[1])):
        color = "white" if cm_recall[i, j] > threshold else "black"
        ax1.text(j, i, labels[i, j], fontsize=10, horizontalalignment="center", color=color)
    ax1.set_ylabel("True label")
    ax1.set_xlabel("Predicted label")
    
    ax2.imshow(cm_precision, interpolation="nearest", cmap=plt.cm.Blues)
    ax2.set_title("Precision matrix")
    ax2.set_xticks(tick_marks, class_names)
    #ax2.set_yticks(tick_marks, class_names)
    # Compute the labels from the normalized confusion matrix.
    # labels = np.around(cm.astype("float"), decimals=2)
    labels = np.around(cm_precision.astype('float') * 100).astype('int')
    # Use white text if squares are dark; otherwise black.
    threshold = cm_precision.max() / 2.0
    for i, j in itertools.product(range(cm_precision.shape[0]), range(cm_precision.shape[1])):
        color = "white" if cm_precision[i, j] > threshold else "black"
        ax2.text(j, i, labels[i, j], fontsize=10, horizontalalignment="center", color=color)
    ax2.set_ylabel("True label")
    ax2.set_xlabel("Predicted label")
    
    return fig

def compute_conf_mat(labels, preds, num_classes, ignore_idx=None):

    if ignore_idx is not None:
        idx = labels != ignore_idx
        preds = preds[idx]
        labels = labels[idx]

    unique_mapping = (labels * num_classes + preds).to(torch.long)
    bins = torch.bincount(unique_mapping, minlength=num_classes**2)
    conf_mat = bins.reshape(num_classes, num_classes)

    return conf_mat
    

class ConfMatLogger(pl.Callback):

    def __init__(self, labels,freq, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.labels = labels
        self.freq = freq

    def on_fit_start(self, writer):

        self.conf_mat = ConfusionMatrix(
            num_classes=len(self.labels),
            normalize=None,
            compute_on_step=False
        )

    def on_validation_batch_end(
            self, writer, batch, batch_idx, dataloader_idx, epoch
    ):
        
        if epoch % self.freq == 0:
            
            #batch = outputs['batch']
            labels = batch['mask'].cpu() # labels shape B,H,W, values in {0, C-1}
            preds = batch['preds'].cpu() 
            self.conf_mat(preds, labels)

    def on_validation_epoch_end(self,epoch, writer):
        
        if epoch % self.freq == 0:

            cm = self.conf_mat.compute().numpy()
            cm_recall = cm/np.sum(cm,axis=1, keepdims=True)
            cm_precision = cm/np.sum(cm,axis=0,keepdims=True)
            
            writer.add_figure(
                "Confusion matrix recall", 
                plot_confusion_matrix(torch.from_numpy(cm_recall), class_names=self.labels), 
                global_step=epoch+1
            )
            writer.add_figure(
                "Confusion matrix precision", 
                plot_confusion_matrix(torch.from_numpy(cm_precision), class_names=self.labels), 
                global_step=epoch+1
            )

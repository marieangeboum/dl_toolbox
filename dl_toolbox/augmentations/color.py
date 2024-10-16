import torch
import torchvision.transforms.functional as F
from .utils import Compose
import  albumentations.augmentations.transforms as A
import  albumentations.augmentations.domain_adaptation as DA



class randomBrightContrast(torch.nn.Module):
    def __init__(self, p=.5, bright_limit = 1, contrast_limit = 1):
        super().__init__()
        self.p = p
        self.bright_limit = bright_limit
        self.contrast_limit = contrast_limit
        #self.albu_random_bright_contrast_aug = A.Compose([brightness_limit=1, contrast_limit=1, p=1.0])
    def forward(self, img , label = None):
        if torch.rand(1).item() < self.p:
            return A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=.5)
        return img, label
 
class ChannelShuffle(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, img, label = None):
        if torch.rand(1).item() < self.p:
            return A.ChannelShuffle(p =0.5)
        return img, label
    

class Gamma(torch.nn.Module):

    def __init__(self, bound=0.5, p=2.5):
        super().__init__()
        self.bounds = (1-bound, 1+bound)
        self.p = p

    def apply(self, img, label=None, factor=1.):

        return F.adjust_gamma(img, factor), label

    def forward(self, img, label=None):

        if torch.rand(1).item() < self.p:
            factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
            return self.apply(img, label, factor)

        return img, label


class Saturation(torch.nn.Module):

    def __init__(self, bound=0.5, p=0.5):
        super().__init__()
        self.bounds = (1-bound, 1+bound)
        self.p = p

    def forward(self, img, label=None):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_saturation(img, factor), label
        return img, label

class Brightness(torch.nn.Module):

    def __init__(self, bound=0.2, p=0.5):
        super().__init__()
        self.bounds = (1-bound, 1+bound)
        self.p = p

    def forward(self, img, label=None):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_brightness(img, factor), label
        return img, label



class Contrast(torch.nn.Module):

    def __init__(self, bound=0.4, p=0.5):
        super().__init__()
        self.bounds = (1-bound, 1+bound)
        self.p = p

    def forward(self, img, label=None):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_contrast(img, factor), label
        return img, label

class Color():

    def __init__(self, p=1, bound=0.3):
        # p has no effect 
        self.bound = bound
        self.color_aug = Compose(
            [
                Saturation(p=1, bound=bound),
                Contrast(p=1, bound=bound),
                Gamma(p=1, bound=bound),
                Brightness(p=1, bound=bound)
            ]
        )

    def __call__(self, image, label=None):
        return self.color_aug(image, label)



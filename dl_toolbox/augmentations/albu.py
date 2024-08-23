import torch
import torchvision.transforms.functional as F
import  albumentations.augmentations.transforms as A
import  albumentations.augmentations.domain_adaptation as DA



class randomBrightContrast():
    def __init__(self, p=0.5, bright_limit = 1, contrast_limit = 1):
        self.p = p
        self.bright_limit = bright_limit
        self.contrast_limit = contrast_limit
        #self.albu_random_bright_contrast_aug = A.Compose([brightness_limit=1, contrast_limit=1, p=1.0])
    def forward(self, img , label = None):
        if torch.rand(1).item() < self.p:
            return A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=.5)
        return img, label
 
class ChannelShuffle():
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, img, label = None):
        if torch.rand(1).item() < self.p:
            return A.ChannelShuffle(p =0.5)
        return img, label
    
# class FourierDomainAdaptation():
#     def __init__(self)
# class randomShadow():
#     def __init__(self, p = 0.5,shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False):
#         super().__init__()
#         self.p = p
#         self.shadow_roi = shadow_roi
#         self.num_shadows_lower = num_shadows_lower
#         self.num_shadows_upper = num_shadows_upper
#         self.shadow_dimension = shadow_dimension
#         self.always_apply = always_apply
#     def forward(self, img, label = None):
#         x_min = torch.rand(1)
#         y_min = torch.rand(1)
#         x_max = torch.rand(1)
#         y_max = torch.rand(1)
        
# class ColorJitter():
#     def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5):
#         super().__init__()
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         self.hue = hue
#         self.always_apply = always_apply 
#         self.p = p
#     def forward(self, img, label = None):
#         if torch.rand(1).item() < self.p:
            
        
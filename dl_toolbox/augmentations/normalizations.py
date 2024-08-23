import torch
import torchvision.transforms.functional as F
<<<<<<< HEAD
# from dl_toolbox.torch_datasets import InriaAustinDs, InriaAllDs, InriaViennaDs
=======
#from dl_toolbox.torch_datasets import InriaDs, InriaAustinDs, InriaAllDs, InriaViennaDs
>>>>>>> origin/MarieAngeBDev

class ImagenetNormalize:

    def __call__(self, img, label=None):

        img = F.normalize(
                img,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        return img, label

# class InriaAustinNormalize:
#     def __call__(self,img,label=None):
#         img = F.normalize(img,mean = InriaAustinDs.stats['mean'], std = InriaAustinDs.stats['mean'])
#         return img, label
    
# class InriaViennaNormalize:
#     def __call__(self,img,label=None):
#         img = F.normalize(img,mean = InriaViennaDs.stats['mean'], std = InriaViennaDs.stats['mean'])
#         return img, label
    
# class InriaAllNormalize:
#     def __call__(self,img,label=None):
#         img = F.normalize(img,mean = InriaAllDs.stats['mean'], std = InriaAllDs.stats['mean'])
#         return img, label
    

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:06:31 2022

@author: marie
"""
import os

from torch.utils.data import Dataset
import torch
from dl_toolbox.utils import get_tiles
from dl_toolbox.torch_datasets.utils import *
from dl_toolbox.torch_datasets.commons import minmax

import rasterio
import imagesize
import numpy as np
from rasterio.windows import Window, bounds, from_bounds, shape
from rasterio.plot import show

from dl_toolbox.utils import MergeLabels, OneHot
import matplotlib.pyplot as plt


class InriaDs(Dataset):

    def __init__(self,image_path,tile,fixed_crops, crop_size,crop_step, img_aug,label_path=None, *args,**kwargs):

        self.image_path = image_path # path to image
        self.label_path = label_path # pth to corresponding label
        self.tile = tile # initializing a tile to be extracted from image
        self.crop_windows = list(get_tiles(
            nols=tile.width,
            nrows=tile.height, 
            size=crop_size,
            step=crop_step,
            row_offset=tile.row_off,
            col_offset=tile.col_off)) if fixed_crops else None # returns a list of tile these are crop extracted from the initial img
        #print("crop windows", self.crop_windows)
        self.crop_size = crop_size # initializing crop size
        self.img_aug = get_transforms(img_aug)


    def read_label(self, label_path, window):
        pass

    def read_image(self, image_path, window):
        pass

    def __len__(self):
        # returns nb of cropped windows
        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        ''' Given index idx this function loads a sample  from the dataset based on index.
            It identifies image'location on disk, converts it to a tensor
        '''

        # in order to visualize what goes IN the network
        # out_path = 'C:/Users/marie/ML/SSL4Remote/output/'
        # output_filename = 'tile_{}-{}.tif' # image file name
        # output_gt_filename = 'tile_gt_{}-{}.tif' # corresponding label file name

        if self.crop_windows:# if self.windows is initialized correctly begin iteration on said list
            window = self.crop_windows[idx]

        else: # otherwise use Winodw method from rasterio module to initilize a window of size cx and cy
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1) # why add those randint ?
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)

        #image = self.read_image(image_path=self.image_path,window=window)  # class inheritance

        ## Not here --> # vizualise the window crops extracted from the input image

        with rasterio.open(self.image_path) as image_file:
            image_rasterio = image_file.read(window=window, out_dtype=np.float32) # read the cropped part of the image
            #print('image', type(image), image.shape)
            #plt.imshow(image.swapaxes(0,-1).astype(np.uint8))
            #plt.title(output_filename.format(int(window.col_off), int(window.row_off)))
            #plt.show()

        # converts image crop into a tensor more precisely returns a contiguous tensor containing the same data as self tensor.
        image = torch.from_numpy(image_rasterio).float().contiguous()

        label = None
        if self.label_path:
            label = self.read_label(
                label_path=self.label_path,
                window=window)
            ## Not here --> # vizualise the window crops extracted from the input image

            with rasterio.open(self.label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
                #print ('label', type(label))
                #show(label)
            # converts label crop into contiguous tensor
            label = torch.from_numpy(label).float().contiguous()

        if self.img_aug is not None:            
            
            #final_image, final_mask = self.img_aug(img = image_rasterio)
                 
                        
            final_image, final_mask = self.img_aug(img=image, label=label)
            
        else:
            final_image, final_mask = image, label
        #print(type(final_image))
        #print(type(window))
        #window = np.array(window)
        #print(type(window),shape(window))
        return {'orig_image':image,
                'orig_mask':label,
                'window':window,
                'image':final_image,
                'mask':final_mask}

class InriaAllDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([103.2342, 108.95195, 100.14193])
    stats['std'] = np.array([51.33931670054912, 46.79573419982041, 44.91756973961484])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
class InriaAllDs_Id(InriaDs):
    stats = {}
    stats['mean'] = np.array([103.2342, 108.95195, 100.14193, 0])
    stats['std'] = np.array([51.33931670054912, 46.79573419982041, 44.91756973961484, 1])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
class InriaAllDs_4C(InriaDs):
    stats = {}
    stats['mean'] = np.array([103.2342, 108.95195, 100.14193, 108.95195])
    stats['std'] = np.array([51.33931670054912, 46.79573419982041, 44.91756973961484, 46.79573419982041])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)        
class InriaAustinDs(InriaDs): 
    stats = {}
    stats['mean'] = np.array([100.94032, 103.52946, 97.66165])
    stats['std'] = np.array([44.22200128195357, 42.98435115090359, 41.719143816688884])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
class InriaAustinViennaDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([111.49419, 111.534805, 104.9858])
    stats['std'] = np.array([52.45912203453931, 47.76822807494897, 46.511076854980935])
    
class InriaAustinChicagoDs(InriaDs):
     stats = {}
     stats['mean'] = np.array([102.17324, 106.39084, 97.57362])
     stats['std'] = np.array([49.58811784858302, 48.39733419399045, 46.77620085468589])
     def __init__(self, *args, **kwargs):
         super().__init__(*args,**kwargs)

class InriaAustinKitsapDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([94.735504, 100.02872, 90.35885])
    stats['std'] = np.array([44.28662718159197, 40.28913937407239, 38.89959993891468])
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

class InriaAustinTyrolDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([101.09303, 109.719696, 103.92902])
    stats['std'] = np.array([47.03375700252017, 44.152273562608926, 40.735093809200116])
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
                    
class InriaViennaDs(InriaDs):
    
    stats = {}
    stats['mean'] = np.array([122.04806, 119.54016, 112.30994])
    stats['std'] = np.array([57.66770860299657, 50.870253477265315, 49.78745482650664])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
class InriaViennaChicagoDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([112.727104, 114.39619, 104.89776])
    stats['std'] = np.array([56.82721534164423, 52.25553889611134, 51.10869727288274])
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
        
class InriaChicagoDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([103.40616, 109.25223, 97.48558])
    stats['std'] = np.array([54.39979130122188, 53.109092671661216, 51.33737867461589])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        

class InriaKitsapDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([88.53069, 96.52798, 83.05605])
    stats['std'] = np.array([43.4744332715294, 37.07108946189805, 34.339593157374054])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
class InriaKitsapViennaDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([105.289375, 108.03407, 97.68299])
    stats['std'] = np.array([53.74615708095641, 45.971866780980136, 45.19896403452171])
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
class InriaKitsapChicagoDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([95.96843, 102.89011, 90.27081])
    stats['std'] = np.array([49.7996177996896, 46.237382896976044, 44.26533158406535])
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
                                                                                                       
class InriaTyrolDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([101.24574, 115.90994, 110.196396])
    stats['std'] = np.array([49.68618074976973, 44.43593357187328, 38.725287488448814])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)  
        
class InriaTyrolViennaDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([111.6469, 117.725044, 111.25317])
    stats['std'] = np.array([54.820844802431644, 47.796060196245, 44.61317686312147])
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
class InriaTyrolChicagoDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([102.32596, 112.581085, 103.84099])
    stats['std'] = np.array([52.10751567379265, 49.077945778724676, 45.912720340474316])
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
class InriaTyrolKitsapDs(InriaDs):
    stats = {}
    stats['mean'] = np.array([94.888214, 106.21896, 96.62622])
    stats['std'] = np.array([47.11465058052803, 42.0514418187218, 39.03303183450496])
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
def main():
    dataset = InriaDs(
        image_path=os.path.join('/data/INRIA/AerialImageDataset/train', 'images/austin1.tif'),
        label_path=os.path.join('/data/INRIA/AerialImageDataset/train', 'gt/austin1.tif'),
        crop_size=256,
        crop_step=256,
        img_aug='imagenet',
        tile=Window(col_off=0, row_off=0, width=400, height=400),
        fixed_crops=True)

    #print(type(dataset))
    #for data in dataset:
#        pass
#    img = plt.imshow(dataset[0]['image'].numpy().transpose(1,2,0))
#        #print(type(data['window']))
#        #print(data, sep = '/n')
#    plt.show()

if __name__ == '__main__':
    main()

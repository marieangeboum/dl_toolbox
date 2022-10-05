# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:06:31 2022

@author: marie
"""
import os
from torch.utils.data import Dataset
import torch
from dl_toolbox.augmentations import get_transforms
from dl_toolbox.torch_datasets.commons import minmax
from dl_toolbox.utils import get_tiles
import rasterio
import imagesize
import numpy as np
from rasterio.windows import Window, bounds, from_bounds
from rasterio.plot import show

from dl_toolbox.utils import MergeLabels, OneHot
import matplotlib.pyplot as plt


class InriaDS(Dataset):

    def __init__(self,image_path,tile,fixed_crops, crop_size,crop_step, img_aug='no',label_path=None, *args,**kwargs):

        self.image_path = image_path # path to image
        self.label_path = label_path # pth to corresponding label
        self.tile = tile # initializing a tile to be extracted from image
        self.crop_windows = list(get_tiles(nols=tile.width,nrows=tile.height, size=crop_size,
            step=crop_step,
            row_offset=tile.row_off,
            col_offset=tile.col_off)) if fixed_crops else None # returns a list of tile these are crop extracted from the initial img
        self.crop_size = crop_size # initializing crop size
        self.img_aug = get_transforms(img_aug)

    def __len__(self):
        # returns nb of cropped windows
        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        ''' Given index idx this function loads a sample  from the dataset baased on index.
            It identifies image'location on disk, converts it to a tensor
        '''

        # in order to visualize what goes IN the network
        out_path = 'C:/Users/marie/ML/SSL4Remote/output/'
        output_filename = 'tile_{}-{}.tif' # image file name
        output_gt_filename = 'tile_gt_{}-{}.tif' # corresponding label file name

        if self.crop_windows:# if self.windows is initialized correctly begin iteration on said list
            window = self.crop_windows[idx]

        else: # otherwise use Winodw method from rasterio module to initilize a window of size cx and cy
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1) # why add those randint ?
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)




        # vizualise the window crops extracted from the input image
        ### IDEA : create a method that suits this purpose ex self.read_image
        with rasterio.open(self.image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32) # read the cropped part of the image
            print('image', type(image), image.shape)

            plt.imshow(image.swapaxes(0,-1).astype(np.uint8))
            plt.title(output_filename.format(int(window.col_off), int(window.row_off)))
            plt.show()

        # converts image crop into a tensor more precisely returns a contiguous tensor containing the same data as self tensor.
        image = torch.from_numpy(image).float().contiguous()

        label = None
        if self.label_path:
            # vizualise the window crops extracted from the input image
            ## IDEA : create a method that suits this purpose ex self.read_label
            with rasterio.open(self.label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
                print ('label', type(label))
                show(label)
            # converts label crop into contiguous tensor
            label = torch.from_numpy(label).float().contiguous()

        if self.img_aug is not None:
            final_image, final_mask = self.img_aug(img=image, label=label)
        else:
            final_image, final_mask = image, label

        return {'orig_image':image,
                'orig_mask':label,
                'image':final_image,
                'window':window,
                'mask':final_mask}


def main():
    dataset = InriaDS(
        image_path='C:/Users/marie/ML/aerialimagelabeling/NEW2-AerialImageDataset/AerialImageDataset/train/images/austin11.tif',
        label_path='C:/Users/marie/ML/aerialimagelabeling/NEW2-AerialImageDataset/AerialImageDataset/train/gt/austin11.tif',
        crop_size=256,
        crop_step=256,
        img_aug='no',
        tile=Window(col_off=0, row_off=0, width=400, height=400),
        fixed_crops=True)

    print(type(dataset))
    for data in dataset:
        print(data['window'].width, data['window'].height)
        #print(type(data['window']))
        #print(data, sep = '/n')

if __name__ == '__main__':
    main()

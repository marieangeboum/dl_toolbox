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


class FlairDs_Id(Dataset):

    def __init__(self, image_path, tile, fixed_crops, crop_size, crop_step, img_aug, task_id,label_path=None, *args, **kwargs):

        self.image_path = image_path  # path to image
        self.label_path = label_path  # pth to corresponding label
        self.tile = tile  # initializing a tile to be extracted from image
        self.task_id = task_id
        self.crop_windows = list(get_tiles(
            nols=tile.width,
            nrows=tile.height,
            size=crop_size,
            step=crop_step,
            row_offset=tile.row_off,
            col_offset=tile.col_off)) if fixed_crops else None  # returns a list of tile these are crop extracted from the initial img
        #print("crop windows", self.crop_windows)
        self.crop_size = crop_size  # initializing crop size
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

        if self.crop_windows:  # if self.windows is initialized correctly begin iteration on said list
            window = self.crop_windows[idx]

        else:  # otherwise use Winodw method from rasterio module to initilize a window of size cx and cy
            # why add those randint ?
            cx = self.tile.col_off + \
                np.random.randint(0, self.tile.width - self.crop_size + 1)
            cy = self.tile.row_off + \
                np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)

        # image = self.read_image(image_path=self.image_path,window=window)  # class inheritance

        # Not here --> # vizualise the window crops extracted from the input image

        with rasterio.open(self.image_path) as image_file:
            # read the cropped part of the image
            image_rasterio = image_file.read(
                window=window, out_dtype=np.float32)
            #print('image', type(image), image.shape)
            
            img_path_strings = self.image_path.split('/')
            domain_pattern = img_path_strings[-4]
            id_domain = domain_pattern.strip('D').replace('_','').lstrip('0')
            id_canal = np.ones(image_rasterio[0].shape)*int(self.task_id)
            new_image = np.stack((image_rasterio[0],image_rasterio[1],image_rasterio[2], id_canal))
            # plt.imshow(image.swapaxes(0,-1).astype(np.uint8))
            #plt.title(output_filename.format(int(window.col_off), int(window.row_off)))
            # plt.show()

        # converts image crop into a tensor more precisely returns a contiguous tensor containing the same data as self tensor.
        image = torch.from_numpy(new_image).float().contiguous()
        
        label = None
        if self.label_path:
            label = self.read_label(
                label_path=self.label_path,
                window=window)
            # Not here --> # vizualise the window crops extracted from the input image

            with rasterio.open(self.label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
                #print ('label', type(label))
                # show(label)
            # converts label crop into contiguous tensor
            label = torch.from_numpy(label).float().contiguous()
            bati_label = torch.tensor(torch.eq(label, 1)*1, dtype=torch.float32)

        if self.img_aug is not None:

            #final_image, final_mask = self.img_aug(img = image_rasterio)

            final_image, final_mask = self.img_aug(img=image, label=bati_label)

        else:
            final_image, final_mask = image, bati_label
        # print(type(final_image))
        # print(type(window))
        #window = np.array(window)
        # print(type(window),shape(window))
        return {'orig_image': image,
                'orig_mask': bati_label,

                'window': window,
                'image': final_image,
                'mask': final_mask}


def main():
    dataset = FlairDs_Id(
        image_path=os.path.join(
            '/data/INRIA/AerialImageDataset/train', 'images/austin1.tif'),
        label_path=os.path.join(
            '/data/INRIA/AerialImageDataset/train', 'gt/austin1.tif'),
        crop_size=256,
        crop_step=256,
        img_aug='imagenet',
        tile=Window(col_off=0, row_off=0, width=400, height=400),
        fixed_crops=True)

    # print(type(dataset))
    # for data in dataset:
#        pass
#    img = plt.imshow(dataset[0]['image'].numpy().transpose(1,2,0))
#        #print(type(data['window']))
#        #print(data, sep = '/n')
#    plt.show()


if __name__ == '__main__':
    main()


# img_path = 'D:/maboum/flair_merged/train/D060_2021/IMG_Z4_UN.tif'

# with rasterio.open(img_path) as image_file:
#     image_rasterio = image_file.read()

import warnings
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
from abc import ABC
from dl_toolbox.utils import get_tiles
import imagesize
from rasterio.rio import insp

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class MultipleImages(Dataset):

    """
    Abstract class that inherits from the standard Torch Dataset abstract class
    and define utilities for remote sensing dataset classes.
    """

    def __init__(self,
                 images_paths=None,
                 crop_size=None,
                 *args,
                 **kwargs
                 ):

        super().__init__()

        self.images_paths = images_paths
        self.crop_size = crop_size

    def minmax(self, img):

        out = np.zeros_like(img).astype(np.float32)
        for i in range(img.shape[0]):
            c = img[i, :, :].min()
            d = img[i, :, :].max()

            t = (img[i, :, :] - c) / (d - c)
            out[i, :, :] = t

        return out.astype(np.float32)

    def __len__(self):

        return len(self.images_paths)

    def __getitem__(self, idx):

        image_path = self.images_paths[idx]

        width, height = imagesize.get(image_path)
        cx = np.random.randint(0, width - self.crop_size + 1)
        cy = np.random.randint(0, height - self.crop_size + 1)
        window = Window(cx, cy, self.crop_size, self.crop_size)

        with rasterio.open(image_path) as image_file:

            image = image_file.read(window=window, out_dtype=np.float32) / 255
            print(insp.stats((image_file, 4)))
        return {'image': image, 'window': window}

class MultipleImagesLabeled(MultipleImages):

    def __init__(self, labels_paths, formatter, *args, **kwargs):

        super(MultipleImagesLabeled, self).__init__(*args, **kwargs)
        assert len(labels_paths) == len(self.images_paths)
        self.labels_paths = labels_paths
        self.labels_formatter = formatter

    def __getitem__(self, idx):

        d = super(MultipleImagesLabeled, self).__getitem__(idx)
        label_path = self.labels_paths[idx]

        with rasterio.open(label_path) as label_file:

            label = label_file.read(window=d['window'], out_dtype=np.float32)

        mask = self.labels_formatter(label)

        return {**d, **{'mask': mask}}
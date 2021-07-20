from dl_toolbox.lightning_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from dl_toolbox.torch_datasets import MultipleImages, MultipleImagesLabeled, miniworld_label_formatter
import glob
from torch.utils.data import ConcatDataset
import numpy as np
from functools import partial


class MiniworldV2(BaseSupervisedDatamodule):

    """
    The train set is the merge of the train folders of all cities in param --cities, the val set is the merge of the
    test folders of all cities in param --cities.
    """

    def __init__(self, cities, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.cities = cities

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--cities", nargs='+', type=str, default=[])

        return parser

    def setup(self, stage=None):

        city_sup_train_sets = []
        city_val_sets = []

        for city in self.cities:

            # Attention on s'appuie sur sorted pour s'assurer de la correspondance image-label: pas très robuste.
            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_x.tif'))
            label_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_y.tif'))
            sup_train_set = MultipleImagesLabeled(
                images_paths=image_list,
                labels_paths=label_list,
                crop_size=self.crop_size,
                formatter=partial(miniworld_label_formatter, city=city)
            )
            city_sup_train_sets.append(sup_train_set)

            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/test/*_x.tif'))
            label_list = sorted(glob.glob(f'{self.data_dir}/{city}/test/*_y.tif'))
            val_set = MultipleImagesLabeled(
                images_paths=image_list,
                labels_paths=label_list,
                crop_size=self.crop_size,
                formatter=partial(miniworld_label_formatter, city=city)
            )
            city_val_sets.append(val_set)

        self.sup_train_set = ConcatDataset(city_sup_train_sets)
        self.val_set = ConcatDataset(city_val_sets)


class MiniworldV2Semisup(MiniworldV2, BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        super(MiniworldV2Semisup, self).setup(stage=stage)

        city_unsup_train_sets = []

        for city in self.cities:

            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_x.tif')) + \
                sorted(glob.glob(f'{self.data_dir}/{city}/test/*_x.tif'))
            unsup_train_set = MultipleImages(
                images_paths=image_list,
                crop_size=self.crop_size
            )
            city_unsup_train_sets.append(unsup_train_set)

        self.unsup_train_set = ConcatDataset(city_unsup_train_sets)
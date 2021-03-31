from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from datasets import IsprsVaihingen, IsprsVaihingenLabeled, IsprsVaihingenUnlabeled
from transforms import MergeLabels
from pl_datamodules import IsprsVaiSup

import albumentations as A
from albumentations.pytorch import ToTensorV2

class IsprsVaiSemisup(IsprsVaiSup):

    def __init__(self, arguments):

        super().__init__(arguments)

        self.unsup_train_transforms = self.sup_train_transforms

    def prepare_data(self, *args, **kwargs):

        # Nothing to write on disk, data is already there, no hard
        # preprocessing necessary
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--nb_im_unsup_train", type=int, default=0)

        return parser

    def setup(self, stage=None):

        labeled_idxs = IsprsVaihingen.labeled_idxs
        np.random.shuffle(labeled_idxs)

        # Here we should use very few labeled images for training...
        val_idxs = labeled_idxs[:self.args.nb_im_val]
        train_idxs = labeled_idxs[-self.args.nb_im_train:]

        self.sup_train_set = IsprsVaihingenLabeled(
            self.data_dir, train_idxs, self.crop_size,
            transforms=None
        )

        self.val_set = IsprsVaihingenLabeled(
            self.data_dir, val_idxs, self.crop_size,
            transforms=None
        )

        # ...but each non validation labeled image can be used without its
        # label for unsupervised training
        unlabeled_idxs = IsprsVaihingen.unlabeled_idxs
        all_unsup_train_idxs = labeled_idxs[self.args.nb_im_val:] + \
                              unlabeled_idxs
        unsup_train_idxs = all_unsup_train_idxs[self.args.nb_im_unsup_train]
        self.unsup_train_set = IsprsVaihingenUnlabeled(
            self.data_dir,
            unsup_train_idxs,
            self.crop_size,
            transforms=None,
        )

    def collate_unlabeled(self, batch):

        transformed_batch = [
            self.unsup_train_transforms(
                image=image
            )
            for image in batch
        ]
        batch = [(elem["image"]) for elem in transformed_batch]

        return default_collate(batch)

    def train_dataloader(self):

        """
        See the supervised dataloader for comments on the need for samplers.
        The semi supervised loader consists in two loaders for labeled and
        unlabeled data.
        """

        sup_train_sampler = RandomSampler(
            data_source=self.sup_train_set,
            replacement=True,
            num_samples=self.nb_pass_per_epoch * len(self.sup_train_set),
        )

        # num_workers should be the number of cpus on the machine.
        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_labeled,
            sampler=sup_train_sampler,
            num_workers=4,
            pin_memory=True,
        )

        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=self.nb_pass_per_epoch * len(self.unsup_train_set),
        )
        # num_workers should be the number of cpus on the machine.
        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_unlabeled,
            sampler=unsup_train_sampler,
            num_workers=4,
            pin_memory=True,
        )

        train_dataloaders = {
            "sup": sup_train_dataloader,
            "unsup": unsup_train_dataloader,
        }

        return train_dataloaders

from functools import partial

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from pl_datamodules import BaseSupervisedDatamodule


class BaseSemisupDatamodule(BaseSupervisedDatamodule):

    def __init__(self, prop_unsup_train, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.prop_unsup_train = prop_unsup_train
        self.unsup_train_set = None

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--prop_unsup_train", type=int, default=1)

        return parser

    def train_dataloader(self):

        """
        See the supervised dataloader for comments on the need for samplers.
        The semi supervised loader consists in two loaders for labeled and
        unlabeled data.
        """

        sup_train_dataloader = super().train_dataloader()

        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.batch_size,
            sampler=unsup_train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.wif
        )

        train_dataloaders = {
            "sup": sup_train_dataloader,
            "unsup": unsup_train_dataloader
        }

        return train_dataloaders
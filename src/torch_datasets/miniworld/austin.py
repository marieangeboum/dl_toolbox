from abc import ABC
from torch_datasets import BaseCity, BaseLabeled, BaseUnlabeled


class Austin(BaseCity, ABC):

    image_size = (3000, 3000)

    def __init__(self, *args, **kwargs):

        super().__init__(city='austin', *args, **kwargs)


class AustinLabeled(Austin, BaseLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class AustinUnlabeled(Austin, BaseUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

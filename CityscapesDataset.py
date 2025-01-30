from os.path import join
from pathlib import Path

from torch.utils.data import Dataset


class KvasirDataset(Dataset):
    _IMG_W = 2048
    _IMG_H = 1024
    _sets = ['train', 'val']

    def __init__(self, dataset_path: str, scale: float = 1):
        self.scale = scale

        self.images_dir = Path(join(dataset_path, 'leftImg8bit'))
        self.mask_dir = Path(join(dataset_path, 'gtFine'))

        # TODO


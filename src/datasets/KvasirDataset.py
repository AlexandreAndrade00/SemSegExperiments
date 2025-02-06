from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms

from utils import load_image


class KvasirDataset(Dataset):
    _IMG_W = 600
    _IMG_H = 500
    NUM_CLASSES = 1

    def __init__(self, dataset_path: str, scale: float = 1):
        self.scale = scale

        self.transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (int(self._IMG_H * scale), int(self._IMG_W * scale)),
                    transforms.InterpolationMode.BICUBIC,
                ),
            ]
        )

        self.target_transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (int(self._IMG_H * scale), int(self._IMG_W * scale)),
                    transforms.InterpolationMode.NEAREST,
                ),
                transforms.Lambda(lambda x: x[0, :, :]),
                transforms.Lambda(lambda x: (x >= 0.5).long()),
            ]
        )

        self.images_dir = Path(join(dataset_path, "images"))
        self.mask_dir = Path(join(dataset_path, "masks"))

        self.ids = [
            splitext(file)[0]
            for file in listdir(self.images_dir)
            if isfile(join(self.images_dir, file)) and not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {dataset_path}, make sure you put your images there"
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert len(img_file) == 1, (
            f"Either no image or multiple images found for the ID {name}: {img_file}"
        )
        assert len(mask_file) == 1, (
            f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        )
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, (
            f"Image and mask {name} should be the same size, but are {img.size} and {mask.size}"
        )

        img = self.transformer(img)
        mask = self.target_transformer(mask)

        return img, mask

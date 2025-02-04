from os.path import join
from os import scandir
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms

from utils import load_image


class CityscapesDataset(Dataset):
    _IMG_W = 2048
    _IMG_H = 1024
    _sets = ["train", "val"]
    NUM_CLASSES = 30

    def __init__(self, dataset_path: str, scale: float = 1):
        self.scale = scale

        self.transformer = (
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        (int(self._IMG_H * scale), int(self._IMG_W * scale)),
                        transforms.InterpolationMode.BICUBIC,
                    ),
                ]
            )
            if scale != 1
            else transforms.ToTensor()
        )

        self.target_transformer = (
            transforms.Compose(
                [
                    transforms.PILToTensor(),
                    transforms.Resize(
                        (int(self._IMG_H * scale), int(self._IMG_W * scale)),
                        transforms.InterpolationMode.NEAREST_EXACT,
                    ),
                    transforms.Lambda(lambda x: x[0, :, :]),
                ]
            )
            if scale != 1
            else transforms.Compose(
                [transforms.PILToTensor(), transforms.Lambda(lambda x: x[0, :, :])]
            )
        )

        self.images_dir = Path(join(dataset_path, "leftImg8bit"))
        self.mask_dir = Path(join(dataset_path, "gtFine"))

        self.ids = []

        for dataset_set in self._sets:
            images_set_dir = self.images_dir.joinpath(dataset_set)

            with scandir(images_set_dir) as it1:
                for city in it1:
                    with scandir(city.path) as it2:
                        city_ids = [
                            join(
                                dataset_set,
                                city.name,
                                "_".join(image.name.split("_")[0:3]),
                            )
                            for image in it2
                        ]

                    self.ids.extend(city_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        image_file = list(self.images_dir.glob(image_id + "_leftImg8bit.png"))
        mask_file = list(self.mask_dir.glob(image_id + "_gtFine_labelIds.png"))

        assert (
            len(image_file) == 1
        ), f"Either no image or multiple images found for the ID {image_id}: {image_file}"
        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {image_id}: {mask_file}"
        mask = load_image(mask_file[0])
        image = load_image(image_file[0])

        assert (
            image.size == mask.size
        ), f"Image and mask {image_id} should be the same size, but are {image.size} and {mask.size}"

        image = self.transformer(image)
        mask = self.target_transformer(mask)

        return image, mask

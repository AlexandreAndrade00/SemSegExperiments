from os.path import join
from typing import ClassVar, List, Self, Tuple
from math import ceil

from torch import Generator, float32
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

from .cityscapes_19 import CityscapesDataset19
from .image_net_1k import ImageNet1K


class DatasetSubsets:
    seed: ClassVar[int] = 1
    data_relative_root_dir: ClassVar[str] = "datasets"
    supported_datasets: ClassVar[List[str]] = ["cityscapes", "image-net-1k"]

    def __init__(
        self,
        train: Dataset,
        exploration_train: Dataset,
        validation: Dataset,
        exploration_validation: Dataset,
        test: Dataset,
        test_batch_size: int,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        dataset_name: str,
    ) -> None:
        self.train = train
        self.exploration_train = exploration_train
        self.validation = validation
        self.exploration_validation = exploration_validation
        self.test = test
        self.test_batch_size = test_batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.dataset_name = dataset_name

    @classmethod
    def from_name(
        cls, name: str, normalise: bool, shape: List[int], augment: bool
    ) -> Self:
        assert len(shape) == 2

        match name:
            case "cityscapes":
                return cls.cityscapes(normalise, shape, augment)
            case _:
                raise ValueError("Dataset not available")

    @classmethod
    def cityscapes(cls, normalise: bool, shape: List[int], augment: bool) -> Self:
        dataset_name = "Cityscapes"

        scaled_img_w = shape[0]
        scaled_img_h = shape[1]

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        transforms_train_list = [
            v2.ToDtype(dtype=float32, scale=True),
        ]

        if augment:
            transforms_train_list += [
                v2.RandomResizedCrop(
                    size=(scaled_img_h, scaled_img_w), scale=(0.125, 1.5)
                ),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(),
            ]

        if normalise:
            transforms_train_list += [
                v2.Normalize(
                    mean=[0.2881, 0.3263, 0.2854], std=[0.1773, 0.1818, 0.1784]
                ),
            ]

        if not augment and not normalise:
            transforms_train_list.append(v2.Resize(size=(scaled_img_h, scaled_img_w)))

        transforms_train = v2.Compose(transforms_train_list)

        transforms_val_test_list = [
            v2.ToDtype(dtype=float32, scale=True),
            v2.Resize(size=(scaled_img_h, scaled_img_w)),
        ]

        if normalise:
            transforms_val_test_list += [
                v2.Normalize(
                    mean=[0.2881, 0.3263, 0.2854], std=[0.1773, 0.1818, 0.1784]
                ),
            ]

        transforms_val_test = v2.Compose(transforms_val_test_list)

        train_data: datasets.VisionDataset = CityscapesDataset19(
            root=data_relative_path,
            split="train",
            mode="fine",
            target_type="semantic",
            transforms=transforms_train,
        )

        train_data_exploring = Subset(train_data, range(ceil(len(train_data) * 0.33)))

        val_data: datasets.VisionDataset = CityscapesDataset19(
            root=data_relative_path,
            split="val",
            mode="fine",
            target_type="semantic",
            transforms=transforms_val_test,
        )

        val_data_exploring = Subset(val_data, range(ceil(len(val_data) * 0.33)))

        test_data = CityscapesDataset19(
            root=data_relative_path,
            split="test",
            mode="fine",
            target_type="semantic",
            transforms=transforms_val_test,
        )

        return cls(
            train_data,
            train_data_exploring,
            val_data,
            val_data_exploring,
            test_data,
            8,
            20,
            (3, scaled_img_h, scaled_img_w),
            dataset_name,
        )

    @classmethod
    def image_net_1k(cls, normalise: bool, shape: List[int], augment: bool) -> Self:
        dataset_name = "ImageNet1K"

        scaled_img_w = shape[0]
        scaled_img_h = shape[1]

        data_relative_path: str = join(cls.data_relative_root_dir, dataset_name)

        transforms_train_list = [
            v2.ToDtype(dtype=float32, scale=True),
        ]

        if augment:
            transforms_train_list += [
                v2.RandomResizedCrop(
                    size=(scaled_img_h, scaled_img_w), scale=(0.125, 1.5)
                ),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(),
            ]
        else:
            transforms_train_list.append(v2.Resize(size=(scaled_img_h, scaled_img_w)))

        if normalise:
            transforms_train_list += [
                v2.Normalize(
                    mean=[0.4813, 0.4574, 0.4077], std=[0.2334, 0.2293, 0.2301]
                ),
            ]

        transforms_train = v2.Compose(transforms_train_list)

        transforms_val_test_list = [
            v2.ToDtype(dtype=float32, scale=True),
            v2.Resize(size=(scaled_img_h, scaled_img_w)),
        ]

        if normalise:
            transforms_val_test_list += [
                v2.Normalize(
                    mean=[0.4813, 0.4574, 0.4077], std=[0.2334, 0.2293, 0.2301]
                ),
            ]

        transforms_val_test = v2.Compose(transforms_val_test_list)

        train_data: datasets.VisionDataset = ImageNet1K(
            root=data_relative_path,
            split="train",
            transforms=transforms_train,
        )

        train_data_exploring = Subset(train_data, range(ceil(len(train_data) * 0.33)))

        val_data: datasets.VisionDataset = ImageNet1K(
            root=data_relative_path,
            split="val",
            transforms=transforms_val_test,
        )

        val_data_exploring = Subset(val_data, range(ceil(len(val_data) * 0.33)))

        test_data = val_data

        return cls(
            train_data,
            train_data_exploring,
            val_data,
            val_data_exploring,
            test_data,
            256,
            1000,
            (3, scaled_img_h, scaled_img_w),
            dataset_name,
        )

    def train_data_loader(
        self, batch_size: int, distributed: bool, exploration: bool
    ) -> DataLoader:
        generator = Generator().manual_seed(self.seed)

        dataset = self.exploration_train if exploration else self.train

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=64,
            persistent_workers=True,
            generator=generator,
            sampler=DistributedSampler(dataset) if distributed else None,
        )

    def validation_data_loader(
        self, batch_size: int, distributed: bool, exploration: bool
    ) -> DataLoader:
        generator = Generator().manual_seed(self.seed)

        dataset = self.exploration_validation if exploration else self.validation

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=48,
            persistent_workers=True,
            generator=generator,
            sampler=DistributedSampler(dataset) if distributed else None,
        )

    def test_data_loader(self, distributed: bool) -> DataLoader:
        generator = Generator().manual_seed(self.seed)

        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,
            generator=generator,
            sampler=DistributedSampler(self.test) if distributed else None,
        )

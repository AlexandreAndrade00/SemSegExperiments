import os

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from torchvision.transforms import InterpolationMode

from data_loading import BasicDataset
from UNet import UNet
from trainer import Trainer
from metrics import iou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scale = 0.5
    img_w = 600
    img_h = 500

    dataset = BasicDataset('/home/alexandre/dev/UNetExperiments/datasets/Kvasir-SEG/images',
                           '/home/alexandre/dev/UNetExperiments/datasets/Kvasir-SEG/masks',
                           transformer=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Resize((int(img_h * scale), int(img_w * scale)),
                                                  InterpolationMode.BICUBIC)]),
                           target_transformer=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Resize((int(img_h * scale), int(img_w * scale)), InterpolationMode.NEAREST),
                                transforms.Lambda(lambda x: x[0, :, :]),
                                transforms.Lambda(lambda x: (x >= 0.5).float())]),
                           )

    # train_set, val_set, test_set = random_split(dataset, [0.6, 0.1, 0.3], generator=torch.Generator().manual_seed(0))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, valid_index) in enumerate(kf.split(dataset)):
        train_set = Subset(dataset, train_index)
        valid_set = Subset(dataset, valid_index)

        train_data_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=os.cpu_count() - 1,
                                       pin_memory=True)
        validation_data_loader = DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=os.cpu_count() - 1,
                                            pin_memory=True)

        model = UNet()

        model = model.to(device)

        loss_fn = BCEWithLogitsLoss()

        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

        trainer = Trainer(training_loader=train_data_loader,
                          validation_loader=validation_data_loader,
                          optimizer=optimizer,
                          loss_fn=loss_fn,
                          model=model,
                          device=device,
                          validation_fn=iou)

        metric = trainer.train()

        print(f"{i} fold metric: {metric}")


if __name__ == '__main__':
    main()

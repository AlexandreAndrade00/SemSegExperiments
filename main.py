import os

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split

from data_loading import BasicDataset
from UNet import UNet
from trainer import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BasicDataset('/home/alexandre/dev/UNetExperiments/datasets/Kvasir-SEG/images',
                           '/home/alexandre/dev/UNetExperiments/datasets/Kvasir-SEG/masks', [0, 255], 0.5)

    train_set, val_set, test_set = random_split(dataset, [0.6, 0.1, 0.3], generator=torch.Generator().manual_seed(0))

    train_data_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=os.cpu_count() - 1,
                                   pin_memory=True)
    validation_data_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=os.cpu_count() - 1,
                                        pin_memory=True)

    model = UNet()

    model = model.to(device)

    loss_fn = BCEWithLogitsLoss()

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    trainer = Trainer(train_data_loader, validation_data_loader, optimizer, loss_fn, model, device)

    trainer.train()


if __name__ == '__main__':
    main()

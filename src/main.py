import os

import torch
from sklearn.model_selection import KFold
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset

from datasets import CityscapesDataset
from nets import PPLiteSeg
from Trainer import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 8

    dataset = CityscapesDataset(
        "/home/alexandre/dev/UNetExperiments/datasets/Cityscapes", scale=0.20
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, valid_index) in enumerate(kf.split(dataset)):
        train_set = Subset(dataset, train_index)
        valid_set = Subset(dataset, valid_index)

        train_data_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
        )
        validation_data_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
        )

        model = PPLiteSeg(num_classes=dataset.NUM_CLASSES, device=device)

        model = model.to(device)

        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

        trainer = Trainer(
            model=model,
            training_loader=train_data_loader,
            validation_loader=validation_data_loader,
            optimizer=optimizer,
            device=device,
            trained_models_path="/home/alexandre/dev/UNetExperiments/trained_models",
        )

        metric = trainer.train()

        print(f"{i} fold metric: {metric}")


if __name__ == "__main__":
    main()

import os

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torch.utils.data import DataLoader, random_split

from src.datasets import CityscapesDataset
from src.nets import PPLiteSeg

from .Trainer import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 8

    dataset = CityscapesDataset(
        "/home/alexandre/dev/SemSegExperiments/datasets/Cityscapes", scale=0.20
    )

    # kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_set, valid_set = random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    # for i, (train_index, valid_index) in enumerate(kf.split(dataset)):
    #     train_set = Subset(dataset, train_index)
    #     valid_set = Subset(dataset, valid_index)

    train_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count() - 1,
    )
    validation_data_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count() - 1,
    )

    model = PPLiteSeg(num_classes=dataset.NUM_CLASSES, device=device)

    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    warmup_epochs: int = 5
    epochs: int = 100

    number_batches: int = len(train_data_loader)
    warmup_iters: int = number_batches * warmup_epochs

    warmup_lr_scheduler = LinearLR(optimizer, total_iters=warmup_iters)

    poly_lr_scheduler = PolynomialLR(
        optimizer,
        total_iters=number_batches * (epochs - warmup_epochs),
        power=0.9,
    )

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, poly_lr_scheduler],
        milestones=[warmup_iters],
    )

    trainer = Trainer(
        model=model,
        training_loader=train_data_loader,
        validation_loader=validation_data_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs + warmup_epochs,
        trained_models_path="/home/alexandre/dev/SemSegExperiments/trained_models",
        lr_scheduler=lr_scheduler,
    )

    metric = trainer.train()

    # print(f"{i} fold metric: {metric}")
    print(f"Metric: {metric}")


if __name__ == "__main__":
    main()

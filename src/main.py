import torch
from torch.optim import SGD
# from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR

from src.datasets.dataset_loader import DatasetSubsets
from src.nets import PPLiteSeg
from src.utils import f1_score_multiclass

from .Trainer import Trainer


def train(model: torch.nn.Module, device: torch.device) -> None:
    batch_size: int = 8

    dataset = DatasetSubsets.cityscapes(False, [1024, 512], True)

    train_data_loader = dataset.train_data_loader(batch_size, False, False)
    validation_data_loader = dataset.validation_data_loader(batch_size, False, False)

    optimizer = SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True
    )

    # warmup_epochs: int = 5
    epochs: int = 100

    # warmup_lr_scheduler = LinearLR(optimizer, total_iters=warmup_epochs)

    # poly_lr_scheduler = PolynomialLR(
    #     optimizer,
    #     total_iters=epochs - warmup_epochs,
    #     power=0.9,
    # )

    # lr_scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_lr_scheduler, poly_lr_scheduler],
    #     milestones=[warmup_epochs],
    # )

    trainer = Trainer(
        model=model,
        training_loader=train_data_loader,
        validation_loader=validation_data_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        trained_models_path="./trained_models",
        # lr_scheduler=lr_scheduler,
        num_classes=dataset.num_classes,
        multiclass_loss_fn=torch.nn.CrossEntropyLoss(),
    )

    metric = trainer.train()

    print(f"Metric: {metric}")


def pretrain(model: torch.nn.Module, device: torch.device) -> None:
    batch_size: int = 128

    dataset = DatasetSubsets.image_net_1k(False, [224, 224], False)

    train_data_loader = dataset.train_data_loader(batch_size, False, False)
    validation_data_loader = dataset.validation_data_loader(batch_size, False, False)

    optimizer = SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True
    )

    # warmup_epochs: int = 5
    epochs: int = 10000

    # warmup_lr_scheduler = LinearLR(optimizer, total_iters=warmup_epochs)

    # poly_lr_scheduler = PolynomialLR(
    #     optimizer,
    #     total_iters=epochs - warmup_epochs,
    #     power=0.9,
    # )

    # lr_scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_lr_scheduler, poly_lr_scheduler],
    #     milestones=[warmup_epochs],
    # )

    trainer = Trainer(
        model=model,
        training_loader=train_data_loader,
        validation_loader=validation_data_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        trained_models_path="./trained_models",
        # lr_scheduler=lr_scheduler,
        num_classes=dataset.num_classes,
        multiclass_loss_fn=torch.nn.CrossEntropyLoss(),
        validation_fn=lambda x, y, z: f1_score_multiclass(x.cpu(), y.cpu()),
        early_stop_patience=100,
    )

    metric = trainer.train()

    print(f"Metric: {metric}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PPLiteSeg(num_classes_train=20, num_classes_pretrain=1000, device=device)

    model = model.to(device)

    model.pre_train()

    pretrain(model, device)

    model.normal_train()

    train(model, device)

    from scripts.test import test
    from scripts.export import export_onnx, export_torch, export_tensorrt

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    dataset = DatasetSubsets.cityscapes(False, [1024, 512], True)

    model_input = next(iter(dataset.validation_data_loader(1, False, False)))[0]

    export_torch(model, "./trained_models", model_input.clone())
    export_tensorrt(model, "./trained_models", model_input.clone())
    export_onnx(model, "./trained_models", model_input.clone())

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    dataset = DatasetSubsets.cityscapes(False, [1024, 512], True)

    metrics = test(
        model,
        test_data_loader=dataset.validation_data_loader(1, False, False),
        num_classes=dataset.num_classes,
        torch_device=device,
        compile=True,
        save_path="./trained_models",
    )

    print(metrics)


if __name__ == "__main__":
    main()

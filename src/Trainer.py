from os.path import join
from pathlib import Path
from typing import Callable, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.utils import calculate_iou


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        optimizer: Optimizer,
        device: torch.device,
        trained_models_path: str,
        num_classes: int,
        epochs: int = 10000,
        early_stop_patience: int = 20,
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
        multiclass_loss_fn: nn.Module = nn.CrossEntropyLoss(),
        validation_fn: Callable[
            [torch.Tensor, torch.Tensor, int], torch.Tensor
        ] = calculate_iou,
        lr_scheduler: LRScheduler | None = None,
    ):
        self.model = model
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.device = device
        self.trained_models_path = trained_models_path
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.multiclass_loss_fn = multiclass_loss_fn
        self.validation_fn = validation_fn
        self.lr_scheduler = lr_scheduler
        self.num_classes: int = num_classes
        self.validation_metrics: Dict["str", float] = {}
        self.early_stop_patience = early_stop_patience

        Path(trained_models_path).mkdir(parents=True, exist_ok=True)

    def train(self) -> float:
        best_metric = 0.0
        early_stop_counting = 0
        self.validation_metrics = {}

        for epoch in range(self.epochs):
            print("EPOCH {}:".format(epoch + 1))

            # train
            self.model.train(True)
            avg_loss = self._train_one_epoch()

            # evaluate
            self.model.eval()
            avg_metric: float = self._val_one_epoch(epoch)

            print("LOSS train {} valid {}".format(avg_loss, avg_metric))

            # Track the best performance, and save the model's state
            if avg_metric > best_metric:
                early_stop_counting = 0

                best_metric = avg_metric

                model_path = join(
                    self.trained_models_path,
                    "model.pt",
                )

                torch.save(self.model.state_dict(), model_path)
            else:
                early_stop_counting += 1

                print(
                    f"EarlyStopping counter: {early_stop_counting} out of {self.early_stop_patience}. Best score {best_metric}, current: {avg_metric}",
                )

                if early_stop_counting >= self.early_stop_patience:
                    return best_metric

        return best_metric

    def _train_one_epoch(self) -> float:
        scaler = torch.amp.GradScaler(self.device.type)

        total_loss = 0.0

        for data in self.training_loader:
            images, true_masks = data

            images = images.to(device=self.device, dtype=torch.float)
            true_masks = true_masks.to(device=self.device, dtype=torch.long)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                # Make predictions for this batch
                masks_pred = self.model(images)

                # Compute the loss and its gradients for c
                if self.num_classes == 1:
                    loss = self.loss_fn(masks_pred.squeeze(1), true_masks.float())
                else:
                    if true_masks.dim() == 4:
                        true_masks = true_masks.squeeze(1)

                    loss = self.multiclass_loss_fn(masks_pred, true_masks)

                    # print(masks_pred)
                    # print(true_masks)

            total_loss += loss.item()

            scaler.scale(loss).backward()

            scaler.step(self.optimizer)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            scaler.update()

        return total_loss / len(self.training_loader)

    def _val_one_epoch(self, epoch: int) -> float:
        running_metric: torch.Tensor = torch.zeros(
            1, device=self.device, dtype=torch.float
        )

        with torch.no_grad():
            for i, vdata in enumerate(self.validation_loader):
                images, true_masks = vdata

                images = images.to(device=self.device, dtype=torch.float)

                true_masks = true_masks.to(device=self.device, dtype=torch.long)

                masks_pred = self.model(images)

                if self.num_classes == 1:
                    assert true_masks.min() >= 0 and true_masks.max() <= 1, (
                        "True mask indices should be in [0, 1]"
                    )

                    masks_pred = (F.sigmoid(masks_pred) > 0.5).float()

                    score = self.validation_fn(true_masks, masks_pred, 1).mean()
                else:
                    assert (
                        true_masks.min() >= 0 and true_masks.max() < self.num_classes
                    ), "True mask indices should be in [0, n_classes["

                    if true_masks.dim() == 4:
                        true_masks = true_masks.squeeze(1)

                    score = self.validation_fn(
                        masks_pred.argmax(dim=1), true_masks, self.num_classes
                    ).mean()

                running_metric += score

        return float((running_metric / len(self.validation_loader)).cpu().item())

    def _wrap_with_profiler(self, fn: Callable) -> None:
        with (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                on_trace_ready=self._profile_trace_handler,
            ) as p,
        ):
            fn()

            p.step()

    def _profile_trace_handler(self, p: torch.profiler.profile) -> None:
        assert self.validation_loader is not None, (
            "The validation data loader should not be None in the validation"
        )

        empty_metrics: Dict[str, float] = {}

        number_samples: int = len(self.validation_loader.dataset)  # type: ignore

        sum_cpu_time: float = 0
        sum_device_time: float = 0
        sum_cpu_memory: float = 0
        sum_gpu_memory: float = 0
        peak_cpu_memory: float = 0
        peak_gpu_memory: float = 0

        assert p.profiler is not None

        for event in p.profiler.function_events:
            sum_cpu_time += event.self_cpu_time_total
            sum_device_time += event.self_device_time_total
            sum_cpu_memory += event.self_cpu_memory_usage
            sum_gpu_memory += event.self_device_memory_usage

            if sum_cpu_memory > peak_cpu_memory:
                peak_cpu_memory = sum_cpu_memory

            if sum_gpu_memory > peak_gpu_memory:
                peak_gpu_memory = sum_gpu_memory

        time_total = sum_cpu_time + sum_device_time

        latency_ms = (time_total / 1000) / number_samples

        throughput_s = number_samples / (time_total / (1000**2))

        peak_cpu_memory_gb = peak_cpu_memory / (1000**3)
        peak_gpu_memory_gb = peak_gpu_memory / (1000**3)

        sum_cpu_memory_gb = sum_cpu_memory / (1000**3)
        sum_gpu_memory_gb = sum_gpu_memory / (1000**3)

        empty_metrics["cpu_peak_memory"] = peak_cpu_memory_gb
        empty_metrics["gpu_peak_memory"] = peak_gpu_memory_gb
        empty_metrics["latency"] = latency_ms
        empty_metrics["throughput"] = throughput_s
        empty_metrics["gpu_allocated_memory"] = sum_gpu_memory_gb
        empty_metrics["cpu_allocated_memory"] = sum_cpu_memory_gb

        self.validation_metrics = empty_metrics

from datetime import datetime
from os.path import join
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils import iou, mean_iou


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        optimizer: Optimizer,
        device: torch.device,
        trained_models_path: str,
        epochs: int = 20,
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
        multiclass_loss_fn: nn.Module = nn.CrossEntropyLoss(),
        validation_fn: Callable[[torch.Tensor, torch.Tensor], float] = iou,
        multiclass_validation_fn: Callable[
            [torch.Tensor, torch.Tensor, int], float
        ] = mean_iou,
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
        self.multiclass_validation_fn = multiclass_validation_fn
        self.epochs = epochs

    def train(self) -> float:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        best_metric = 0.0

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
                best_metric = avg_metric

                model_path = join(
                    self.trained_models_path,
                    "{}_{}_{}".format(self.model.name, timestamp, epoch),
                )

                torch.save(self.model.state_dict(), model_path)

        return best_metric

    def _train_one_epoch(self):
        total_loss = 0.0

        for data in self.training_loader:
            images, true_masks = data

            images = images.to(
                device=self.device, dtype=torch.float, memory_format=torch.channels_last
            )
            true_masks = true_masks.to(device=self.device, dtype=torch.long)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            masks_pred = self.model(images)

            # Compute the loss and its gradients for c
            if self.model.num_classes == 1:
                loss = self.loss_fn(masks_pred.squeeze(1), true_masks.float())
            else:
                loss = self.multiclass_loss_fn(masks_pred, true_masks)

            total_loss += loss.item()

            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

        return total_loss / self.training_loader.__len__()

    def _val_one_epoch(self, epoch: int) -> float:
        running_metric: float = 0.0

        with (
            torch.no_grad(),
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                on_trace_ready=lambda profiler: self._profile_trace_handler(
                    profiler, epoch
                ),
            ) as p,
        ):
            for i, vdata in enumerate(self.validation_loader):
                images, true_masks = vdata

                images = images.to(
                    device=self.device,
                    dtype=torch.float,
                    memory_format=torch.channels_last,
                )

                true_masks = true_masks.to(device=self.device, dtype=torch.long)

                masks_pred = self.model(images)

                if self.model.num_classes == 1:
                    assert true_masks.min() >= 0 and true_masks.max() <= 1, (
                        "True mask indices should be in [0, 1]"
                    )

                    masks_pred = (F.sigmoid(masks_pred) > 0.5).float()

                    score = self.validation_fn(true_masks, masks_pred)
                else:
                    assert (
                        true_masks.min() >= 0
                        and true_masks.max() < self.model.num_classes
                    ), "True mask indices should be in [0, n_classes["

                    score = self.multiclass_validation_fn(
                        true_masks, masks_pred.argmax(dim=1), self.model.num_classes
                    )

                running_metric += score

                p.step()

        return running_metric / len(self.validation_loader)

    def _profile_trace_handler(self, p, epoch):
        number_samples = len(self.validation_loader.dataset)

        sum_cpu_time = 0
        sum_device_time = 0
        sum_cpu_memory = 0
        sum_gpu_memory = 0
        peak_cpu_memory = 0
        peak_gpu_memory = 0

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

        # GPU dependent, necessary to install specific packages
        # print(torch.cuda.power_draw())
        print(f"Latency(ms): {latency_ms}")
        print(f"Throughput(FPS): {throughput_s}")
        print(f"Peak GPU Memory(GB): {peak_gpu_memory_gb}")
        print(f"Peak CPU Memory(GB): {peak_cpu_memory_gb}")

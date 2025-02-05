from datetime import datetime

import torch


class Trainer:
    def __init__(
        self,
        training_loader,
        validation_loader,
        optimizer,
        loss_fn,
        model,
        device,
        validation_fn,
        model_name: str,
    ):
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = model
        self.device = device
        self.validation_fn = validation_fn
        self.model_name = model_name

    def train_one_epoch(self):
        total_loss = 0.0

        for data in self.training_loader:
            images, true_masks = data

            images = images.to(
                device=self.device, dtype=torch.float, memory_format=torch.channels_last
            )
            true_masks = true_masks.to(device=self.device, dtype=torch.float)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            masks_pred = self.model(images)

            # Compute the loss and its gradients for c
            if masks_pred.size()[1] == 3:
                loss = self.loss_fn(masks_pred, true_masks)
            else:
                loss = self.loss_fn(masks_pred.squeeze(1), true_masks)

            total_loss += loss.item()

            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

        return total_loss / self.training_loader.__len__()

    def train(self) -> float:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        epoch_number = 0

        EPOCHS = 20

        best_metric = 0.0

        for epoch in range(EPOCHS):
            print("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch()

            running_metric = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
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
                    true_masks = true_masks.to(device=self.device, dtype=torch.float)

                    masks_pred = self.model(images)

                    if masks_pred.size()[1] == 3:
                        score = self.validation_fn(true_masks, masks_pred)
                    else:
                        score = self.validation_fn(true_masks, masks_pred.squeeze(1))

                    running_metric += score

                    p.step()

            avg_metric = running_metric / (i + 1)
            print("LOSS train {} valid {}".format(avg_loss, avg_metric))

            # Track the best performance, and save the model's state
            if avg_metric > best_metric:
                best_metric = avg_metric
                model_path = "../trained_models/{}_{}_{}".format(
                    self.model_name, timestamp, epoch_number
                )
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

        return best_metric

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

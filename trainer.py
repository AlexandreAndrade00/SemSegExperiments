from datetime import datetime

import torch
import time


class Trainer:
    def __init__(self, training_loader, validation_loader, optimizer, loss_fn, model, device, validation_fn):
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = model
        self.device = device
        self.validation_fn = validation_fn

    def train_one_epoch(self):
        total_loss = 0.

        for data in self.training_loader:
            images, true_masks = data

            images = images.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=self.device, dtype=torch.long)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            masks_pred = self.model(images)

            # Compute the loss and its gradients for c
            loss = self.loss_fn(masks_pred.squeeze(1), true_masks.float())

            total_loss += loss.item()

            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

        return total_loss / self.training_loader.__len__()

    def train(self) -> float:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        epoch_number = 0

        EPOCHS = 20

        best_metric = 0.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch()

            running_metric = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad(), torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ], profile_memory=True, on_trace_ready=self._profile_trace_handler) as p:

                for i, vdata in enumerate(self.validation_loader):
                    images, true_masks = vdata

                    images = images.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=self.device, dtype=torch.long)

                    masks_pred = self.model(images)

                    score = self.validation_fn(true_masks, masks_pred.squeeze(1))

                    running_metric += score

                    p.step()

            avg_metric = running_metric / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_metric))

            # Track the best performance, and save the model's state
            if avg_metric > best_metric:
                best_metric = avg_metric
                model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

        return best_metric

    def _profile_trace_handler(self, p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)

        number_samples = len(self.validation_loader)

        # TODO: throughput, latency

        print(output)
        print(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

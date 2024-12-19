from datetime import datetime

import torch


class Trainer:
    def __init__(self, training_loader, validation_loader, optimizer, loss_fn, model, device):
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = model
        self.device = device

    def train_one_epoch(self):
        total_loss = 0.

        for i, data in enumerate(self.training_loader):
            images, true_masks = data['image'], data['mask']

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

    def train(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        epoch_number = 0

        EPOCHS = 5

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch()

            running_v_loss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    images, true_masks = vdata['image'], vdata['mask']

                    images = images.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=self.device, dtype=torch.long)

                    masks_pred = self.model(images)

                    loss = self.loss_fn(masks_pred.squeeze(1), true_masks.float())

                    running_v_loss += loss

            avg_v_loss = running_v_loss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_v_loss))

            # Track the best performance, and save the model's state
            if avg_v_loss < best_vloss:
                best_vloss = avg_v_loss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

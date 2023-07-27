import torch
import time
from sklearn.metrics import fbeta_score
from torch import autocast
import torch.distributed

from utils.logger import get_logger, DEBUG
from utils.dist import is_distributed

logger = get_logger('train')


def train(model, optimizer, lr_scheduler, criterion, scaler, train_loader, epochs, device, use_amp, amp_dtype,
          epoch):

    # Run an epoch
    start_epoch_time = time.time()
    start_iteration_time = time.time()
    iterations_samples_per_second = []

    if is_distributed():
        train_loader.sampler.set_epoch(epoch)

    train_epoch_loss = 0.0
    train_epoch_fbeta = 0.0

    model.train()
    percentages = [int(len(train_loader) * i / 100) for i in range(0, 100, 10)][1:]
    for i, (sub_volumes, ink_labels) in enumerate(train_loader):
        optimizer.zero_grad()

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):

            outputs = model(sub_volumes.to(device))
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, ink_labels.to(device).float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        train_epoch_loss += loss.item()
        outputs = torch.sigmoid(outputs.detach())
        ink_predictions = torch.where(outputs.detach() > 0.5, 1, 0).cpu()

        ink_labels = ink_labels.reshape(-1)
        ink_predictions = ink_predictions.reshape(-1)
        ink_labels = torch.where(ink_labels == 2, 0, ink_labels)
        ink_predictions = torch.where(ink_predictions == 2, 0, ink_predictions)

        train_epoch_fbeta += fbeta_score(ink_labels.numpy(), ink_predictions.numpy(), beta=0.5, zero_division=0)

        if i in percentages:
            processed_images = (i + 1) * train_loader.batch_size
            samples_per_second = int(processed_images / (time.time() - start_iteration_time))

            logger.info(f"Epoch [{epoch}]/[{epochs}]. Train Step [{i}][{len(train_loader)}]. "
                        f"Train Loss: {train_epoch_loss / i}. Train Fbeta: {train_epoch_fbeta / i}. "
                        f"Samples per second: {samples_per_second}")

            iterations_samples_per_second.append(samples_per_second)

        if DEBUG and i % 6000 == 6000 - 1:
            logger.info(f'DEBUG mode: breaking')
            break

        if is_distributed():
            torch.distributed.barrier()

    return train_epoch_loss / len(train_loader), train_epoch_fbeta / len(train_loader), start_epoch_time

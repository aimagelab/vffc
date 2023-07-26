import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import save_image
from pathlib import Path

from .sub_volume_dataset import  WhatToPredict, Classes

from utils.logger import get_logger

logger = get_logger(__file__)


def detect_papyrus_ink(model, dataset, batch_size, threshold, load_labels, device, epoch,
                       save_path=None):

    logger.info(f'Performing test detection on {dataset.image_id}...')
    model.eval()

    test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    output = torch.full(size=dataset.mask.shape, fill_value=0).float()
    predicted_locations = torch.full(size=dataset.mask.shape, fill_value=0).float()

    percentages = [int(len(test_loader) * i / 100) for i in range(0, 100, 20)][1:]
    with torch.no_grad():
        for i, (sub_volumes, _) in enumerate(test_loader):

            outputs = model(sub_volumes.to(device))

            inner_patch_start = dataset.buffer - dataset.inner_patch_buffer
            inner_patch_end = dataset.buffer + dataset.inner_patch_buffer
            outputs = outputs[:, :, inner_patch_start:inner_patch_end, inner_patch_start:inner_patch_end]
            ink_predictions_probabilities = outputs.permute(0, 2, 3, 1)
            if dataset.classes == Classes.PAPYRUS_INK_BACKGROUND:
                ink_predictions_probabilities = nn.functional.softmax(ink_predictions_probabilities, dim=-1)
                ink_predictions_probabilities = ink_predictions_probabilities[:, :, :, 1].detach().cpu()
            else:
                ink_predictions_probabilities = torch.sigmoid(ink_predictions_probabilities.detach())
                ink_predictions_probabilities = ink_predictions_probabilities[:, :, :, 0].detach().cpu()

            for j, value in enumerate(ink_predictions_probabilities):
                patch_num = i * batch_size + j
                if patch_num < len(dataset.valid_pixels):
                    y, x = dataset.valid_pixels[patch_num] + dataset.padding
                    if dataset.what_to_predict == WhatToPredict.SINGLE_PIXEL:
                        output[y, x] = value.item()
                    else:
                        y_start_out = max(0, y - dataset.inner_patch_buffer)
                        y_start_value = y_start_out - (y - dataset.inner_patch_buffer)
                        y_end_out = min(output.shape[0], y + dataset.inner_patch_buffer)
                        y_end_value = value.shape[0] - ((y + dataset.inner_patch_buffer) - y_end_out)

                        x_start_out = max(0, x - dataset.inner_patch_buffer)
                        x_start_value = x_start_out - (x - dataset.inner_patch_buffer)
                        x_end_out = min(output.shape[1], x + dataset.inner_patch_buffer)
                        x_end_value = value.shape[1] - ((x + dataset.inner_patch_buffer) - x_end_out)

                        output_y_slice = slice(y_start_out, y_end_out, 1)
                        output_x_slice = slice(x_start_out, x_end_out, 1)
                        value_y_slice = slice(y_start_value, y_end_value, 1)
                        value_x_slice = slice(x_start_value, x_end_value, 1)
                        output[output_y_slice, output_x_slice] += value[value_y_slice, value_x_slice]
                        predicted_locations[output_y_slice, output_x_slice] += 1

            if i in percentages:
                logger.info(f'Processed [{i * batch_size}]/[{len(dataset)}] sub-volumes')

            a = False
            if a:
                break

    output = torch.where(dataset.mask == 1, output, 0)
    predicted_locations = torch.where(dataset.mask == 1, predicted_locations, 0)
    if (predicted_locations - dataset.mask).sum() != 0:
        logger.exception(f'Predicted locations and output are not the same: {predicted_locations - output}')
        raise Exception('Predicted locations and output are not the same')

    output = output[dataset.padding:-dataset.padding, dataset.padding:-dataset.padding]

    output_threshold = output.gt(threshold).float().cpu()
    if save_path:
        save_image(output.cpu(), Path(save_path) / f'output_{dataset.image_id}_{epoch:03d}.png')
        save_image(output_threshold, Path(save_path) / f'output_th_{dataset.image_id}_{epoch:03d}.png')

    result = {
        'papyrus_id': dataset.image_id,
        'detection': output_threshold.numpy(),
        'middle_papyrus_slice': dataset.middle_slice,
        'labels': dataset.un_padded_labels if load_labels else '',
        'class_labels': dataset.class_labels if load_labels else ''
    }

    return result

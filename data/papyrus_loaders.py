import numpy as np
from pathlib import Path
import PIL.Image as Image

from utils.logger import get_logger, DEBUG

logger = get_logger(__file__)


def load_papyruses(train_data_paths, load_patch_size, load_z_start, load_z_size,
                   patch_size, z_start, z_size, load_labels=True):

    papyruses = {}

    for i, papyrus_path in enumerate(train_data_paths):

        papyrus_id = papyrus_path.stem.split('_')[0]

        if DEBUG and (papyrus_id == '2' or papyrus_id == '3'):
            continue
        logger.info(f'Loading {papyrus_path}')

        papyrus = load_papyrus(
            image_path=papyrus_path,
            load_patch_size=load_patch_size,
            load_z_start=load_z_start,
            load_z_size=load_z_size,
            patch_size=patch_size,
            z_start=z_start,
            z_size=z_size,
            load_labels=load_labels,
        )

        papyrus['split'] = 'eval' if papyrus_id == 1 else 'train'
        logger.info(f'Done loading {papyrus_path}. Assigned to split {papyrus["split"]}.')

        papyruses[papyrus_path.stem] = papyrus

    return papyruses


def load_papyrus(image_path: Path, load_patch_size: int, load_z_start: int, load_z_size: int, patch_size: int,
                 z_start: int, z_size: int, load_labels: bool):

    assert patch_size % 2 == 0, 'Patch size must be even'
    padding = load_patch_size
    images_paths = sorted(list((image_path / 'surface_volume').iterdir()))[load_z_start:load_z_start + load_z_size]
    images = [np.array(Image.open(image_path)) for image_path in images_paths]
    image_stack = np.stack(images, axis=0)
    image_stack = np.pad(image_stack, ((0, 0), (padding, padding), (padding, padding)))
    mask = np.array(Image.open(str(image_path / 'mask.png'))).astype(np.uint8)

    labels = None
    if load_labels:
        labels = np.array(Image.open(str(image_path / 'inklabels.png'))).astype(np.uint8)

    papyrus = {
        'papyrus_path': image_path,
        'load_patch_size': load_patch_size,
        'load_z_start': load_z_start,
        'load_z_size': load_z_size,
        'patch_size': patch_size,
        'padding': padding,
        'z_start': z_start,
        'z_size': z_size,
        'load_labels': load_labels,
        'image_stack': image_stack,
        'mask': mask,
        'labels': labels
    }

    return papyrus

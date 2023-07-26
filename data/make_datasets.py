import torch.utils.data as data

from data.sub_volume_dataset import GridPatchedSubVolumeDataset, GridFullPatchedSubVolumeDataset
from utils.logger import get_logger

logger = get_logger(__file__)


def make_train_dataset(papyruses, load_patch_size, load_z_start, load_z_size, patch_size, z_start, z_size,
                       train_stride, train_transforms=None):
    train_datasets = []

    for i, (papyrus_name, papyrus) in enumerate(papyruses.items()):

        papyrus_path = papyrus['papyrus_path']

        if papyrus['split'] == 'eval':
            logger.info(f'Skipping eval papyrus {papyrus_name}')
            continue

        train_datasets.append(
            GridPatchedSubVolumeDataset(
                image_path=papyrus_path,
                load_patch_size=load_patch_size,
                load_z_start=load_z_start,
                load_z_size=load_z_size,
                patch_size=patch_size,
                z_start=z_start,
                z_size=z_size,
                load_labels=True,
                transform=train_transforms,
                stride=train_stride,
                num_dims=4,
                loaded_papyrus=papyrus,
                train=True))

    train_dataset = data.ConcatDataset(train_datasets)
    return train_dataset


def make_eval_dataset(papyruses, patch_size, z_start, z_size, eval_stride, eval_transforms=None):
    eval_dataset = None
    for i, (papyrus_name, papyrus) in enumerate(papyruses.items()):
        if papyrus['split'] == 'train':
            continue

        eval_dataset = GridPatchedSubVolumeDataset(
            image_path=papyrus['papyrus_path'],
            patch_size=patch_size,
            z_start=z_start,
            z_size=z_size,
            load_labels=True,
            transform=eval_transforms,
            stride=eval_stride,
            num_dims=3,
            loaded_papyrus=papyrus,
            train=False)

    return eval_dataset


def make_test_dataset(papyruses, patch_size, z_start, z_size, test_stride, test_transforms=None):

    test_dataset = None
    for i, (papyrus_name, papyrus) in enumerate(papyruses.items()):
        if papyrus['split'] == 'train':
            continue

        test_dataset = GridFullPatchedSubVolumeDataset(
            image_path=papyrus['papyrus_path'],
            patch_size=patch_size,
            z_start=z_start,
            z_size=z_size,
            load_labels=True,
            transform=test_transforms,
            stride=test_stride,
            num_dims=3,
            loaded_papyrus=papyrus,
            train=False)

    return test_dataset

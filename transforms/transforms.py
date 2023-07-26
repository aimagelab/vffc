import torch.utils.data
import torch
import math
import albumentations as A
from transforms.RandomMaskSafeCrop import RandomMaskSafeCrop
from transforms.ZSwing import ZSwing


def get_data_load_settings(args, load_patch_size, load_z_start, load_z_size):
    transforms_settings = {
        'enable_train_augmentations': args.enable_train_augmentations,
        'dihedral_prob': args.dihedral_prob,
        'random_crop_prob': args.random_crop_prob,
        'xy_swing_prob': args.random_crop_prob,
        'xy_swing': 128,
        'channel_dropout_prob': args.channel_dropout_prob,
        'channel_dropout_max_factor': args.channel_dropout_max_factor,
        'z_swing_prob': args.random_crop_prob,
        'z_swing': 8,
    }
    load_settings = {'load_patch_size': load_patch_size, 'load_z_start': load_z_start, 'load_z_size': load_z_size}

    if not transforms_settings['enable_train_augmentations']:
        return transforms_settings, load_settings

    if transforms_settings['xy_swing_prob'] > 0:
        load_settings['load_patch_size'] += 2 * transforms_settings['xy_swing']
    if transforms_settings['z_swing_prob'] > 0:
        z_swing = transforms_settings['z_swing']
        load_settings['load_z_size'] += 2 * z_swing
        load_settings['load_z_start'] = load_z_start - z_swing
        assert load_settings['load_z_start'] >= 0, 'z_swing is too big'
        assert load_settings['load_z_start'] + load_settings['load_z_size'] <= 65, 'z_swing is too big'

    assert load_z_size <= 65, 'load_z_size must be <= 65, try setting less augmentations on z'

    return transforms_settings, load_settings


def get_train_transforms(transforms_settings, patch_size, z_size):
    if not transforms_settings['enable_train_augmentations']:
        return None

    train_transforms = []

    if transforms_settings['dihedral_prob'] > 0:
        dihedral_transforms = A.Compose([
            A.HorizontalFlip(p=transforms_settings['dihedral_prob']),
            A.VerticalFlip(p=transforms_settings['dihedral_prob']),
            A.RandomRotate90(p=transforms_settings['dihedral_prob']),
            A.Transpose(p=transforms_settings['dihedral_prob'])
        ], p=0.8)
        train_transforms.append(dihedral_transforms)
    if transforms_settings['xy_swing_prob'] > 0:
        train_transforms.append(RandomMaskSafeCrop(patch_size=patch_size, p=transforms_settings['xy_swing_prob']))
    if transforms_settings['z_swing_prob'] > 0:
        train_transforms.append(ZSwing(z_size=z_size, p=transforms_settings['z_swing_prob']))
    if transforms_settings['channel_dropout_prob'] > 0:
        channel_drop_max = round(z_size * transforms_settings['channel_dropout_max_factor'])
        channel_drop_range = (1, channel_drop_max)
        train_transforms.append(
            A.ChannelDropout(channel_drop_range=channel_drop_range, p=transforms_settings['channel_dropout_prob']))

    train_transforms = A.Compose(train_transforms)

    return train_transforms


def get_eval_transforms(transforms_settings, z_size):
    if not transforms_settings['enable_train_augmentations']:
        return None

    eval_transforms = []
    if transforms_settings['z_swing_prob'] > 0:
        eval_transforms.append(ZSwing(z_size=z_size, p=0.))

    eval_transforms = A.Compose(eval_transforms)
    return eval_transforms
#
#
# def get_test_transforms():
#     test_transforms = transforms.Compose([
#         transforms.ToTensor()
#     ])
#
#     return test_transforms

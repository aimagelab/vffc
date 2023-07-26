import torch.nn as nn


def normalization(x):
    dim = list(range(1, x.ndim))
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + 1e-9)


def conv_nd_class(dims):
    if dims == 2:
        return nn.Conv2d
    if dims == 3:
        return nn.Conv3d
    raise ValueError(f'Unsupported dimension {dims}')


def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'Unsupported dimension {dims}')


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'Unsupported dimension {dims}')


def max_pool_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.MaxPool2d(*args, **kwargs)
    if dims == 3:
        return nn.MaxPool3d(*args, **kwargs)
    raise ValueError(f'Unsupported dimension {dims}')


def norm_nd_class(dims):
    if dims == 2:
        return nn.BatchNorm2d
    if dims == 3:
        return nn.BatchNorm3d
    raise ValueError(f'Unsupported dimension {dims}')


def norm_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.BatchNorm2d(*args, **kwargs)
    if dims == 3:
        return nn.BatchNorm3d(*args, **kwargs)
    raise ValueError(f'Unsupported dimension {dims}')

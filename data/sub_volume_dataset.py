import numpy as np
import torch.utils.data as data
from pathlib import Path
import torch
from typing import Callable, Dict,  Optional
import enum
from utils.logger import get_logger

from .papyrus_loaders import load_papyrus

logger = get_logger(__file__)


class FragmentLabels(enum.IntEnum):
    PAPYRUS = 0
    INK = 1
    BACKGROUND = 2


class WhatToPredict(enum.IntEnum):
    SINGLE_PIXEL = 0
    PATCH = 1


class Classes(enum.IntEnum):
    PAPYRUS_INK = 0
    PAPYRUS_INK_BACKGROUND = 1


class SubVolumeDataset(data.Dataset):
    def __init__(
            self,
            image_path: Path,
            patch_size: int,
            z_start: int,
            z_size: int,
            load_labels: bool,
            num_dims: int,
            loaded_papyrus: Optional[Dict] = None,
            transform: Optional[Callable] = None,
            load_patch_size: Optional[int] = None,
            load_z_start: Optional[int] = None,
            load_z_size: Optional[int] = None,
            train: bool = True
    ):

        assert patch_size % 2 == 0, 'Patch size must be even'
        image_path = Path(image_path)
        self.image_id = image_path.stem
        self.classes = Classes.PAPYRUS_INK
        self.num_dims = num_dims
        self.class_labels = {
            FragmentLabels.PAPYRUS: 'papyrus', FragmentLabels.INK: 'ink'} if self.classes == Classes.PAPYRUS_INK else {
            FragmentLabels.PAPYRUS: 'papyrus', FragmentLabels.INK: 'ink', FragmentLabels.BACKGROUND: 'background'}

        if loaded_papyrus is None:
            papyrus = load_papyrus(
                image_path=image_path,
                load_patch_size=load_patch_size if load_patch_size is not None else patch_size,
                load_z_start=load_z_start if load_z_start is not None else z_start,
                load_z_size=load_z_size if load_z_size is not None else z_size,
                patch_size=patch_size,
                z_start=z_start,
                z_size=z_size,
                load_labels=load_labels)

        else:
            papyrus = loaded_papyrus

        self.image_id = papyrus['papyrus_path'].stem
        self.image_stack = papyrus['image_stack']
        self.mask = papyrus['mask']
        self.patch_size = papyrus['patch_size']
        self.load_patch_size = papyrus['load_patch_size']
        self.load_z_start = papyrus['load_z_start']
        self.load_z_size = papyrus['load_z_size']
        self.z_start = papyrus['z_start']
        self.z_size = papyrus['z_size']
        self.padding = papyrus['padding']
        self.load_labels = papyrus['load_labels']

        self.image_height, self.image_width = self.mask.shape

        if self.load_labels:
            self.labels = papyrus['labels']
            self.labels = np.where(self.mask == 0, FragmentLabels.BACKGROUND, self.labels)
            labels_pad = ((self.padding, self.padding), (self.padding, self.padding))
            self.labels = np.pad(self.labels, labels_pad, mode='constant', constant_values=FragmentLabels.BACKGROUND)

        self.valid_pixels = np.argwhere(self.mask).astype(np.uint16)
        mask_pad = ((self.padding, self.padding), (self.padding, self.padding))
        self.mask = np.pad(self.mask, mask_pad, mode='constant', constant_values=0)
        self.mask = torch.from_numpy(self.mask)
        self.buffer = self.load_patch_size // 2 if train else self.patch_size // 2
        self.transform = transform

    @property
    def un_padded_labels(self):
        labels = self.labels[self.padding:-self.padding, self.padding:-self.padding]
        labels = np.where(labels == FragmentLabels.BACKGROUND, FragmentLabels.PAPYRUS, labels)
        return labels

    @property
    def un_padded_mask(self):
        return self.mask[self.padding:-self.padding, self.padding:-self.padding]

    @property
    def middle_slice(self):
        middle_slice = self.image_stack[int(self.z_size // 2), self.padding:-self.padding, self.padding:-self.padding]
        return middle_slice.astype(np.float32) / 65535

    def __len__(self):
        return len(self.valid_pixels)

    def __getitem__(self, index):
        y, x = self.valid_pixels[index] + self.padding
        sub_volume = self.image_stack[:, y - self.buffer:y + self.buffer, x - self.buffer:x + self.buffer]
        if self.num_dims == 5:
            sub_volume = np.expand_dims(sub_volume, 0)
        sub_volume = sub_volume.astype(np.float32) / 65535

        if self.load_labels:
            labels = self.labels[y - self.buffer:y + self.buffer, x - self.buffer:x + self.buffer]
            if self.num_dims == 5:
                labels = np.expand_dims(labels, 0)
        else:
            labels = np.array([-1])

        if self.transform:
            sub_volume = sub_volume.transpose(1, 2, 0)  # CHW -> HWC
            if self.load_labels:
                result = self.transform(image=sub_volume, mask=labels)
                labels = result['mask']
            else:
                result = self.transform(image=sub_volume)
            sub_volume = result['image']
            sub_volume = sub_volume.transpose(2, 0, 1)  # HWC -> CHW

        labels = np.where(labels == FragmentLabels.BACKGROUND, FragmentLabels.PAPYRUS, labels)

        return torch.from_numpy(sub_volume), torch.from_numpy(labels.astype(np.int64))


class GridPatchedSubVolumeDataset(SubVolumeDataset):
    def __init__(
            self,
            image_path: Path,
            patch_size: int,
            z_start: int,
            z_size: int,
            load_labels: bool,
            transform: Optional[Callable] = None,
            stride: int = 1,
            num_dims: int = 5,
            loaded_papyrus: Optional[Dict] = None,
            load_patch_size: Optional[int] = None,
            load_z_start: Optional[int] = None,
            load_z_size: Optional[int] = None,
            train: bool = True
    ):

        super().__init__(
            image_path=image_path,
            patch_size=patch_size,
            z_start=z_start,
            z_size=z_size,
            load_labels=load_labels,
            num_dims=num_dims,
            loaded_papyrus=loaded_papyrus,
            transform=transform,
            load_patch_size=load_patch_size,
            load_z_start=load_z_start,
            load_z_size=load_z_size,
            train=train)

        self.stride = stride
        height_coordinates = [i for i in range(0, self.image_height, self.stride)]
        width_coordinates = [i for i in range(0, self.image_width, self.stride)]
        width_v, height_v = np.meshgrid(width_coordinates, height_coordinates, indexing='ij')
        grid_pixels = np.stack([height_v, width_v], axis=2).reshape(
            (len(height_coordinates) * len(width_coordinates), 2))
        grid_pixels = grid_pixels.astype(np.uint16)

        # Check if each patch contains valid pixels
        valid_pixels = []
        for pixel in grid_pixels:
            y, x = pixel + self.padding
            slice_y = slice(y - self.buffer, y + self.buffer, 1)
            slice_x = slice(x - self.buffer, x + self.buffer, 1)
            if (self.mask[slice_y, slice_x]).sum() != 0:
                valid_pixels.append(pixel)

        self.valid_pixels = np.array(valid_pixels)


def compute_possible_strides(image_height, image_width, stride_start, stride_end, stride_step):
    possible_strides = []
    for stride in range(stride_start, stride_end + 1, stride_step):
        inner_patch_buffer = stride / 2
        possible_h = (image_height - inner_patch_buffer) % stride < inner_patch_buffer
        possible_w = (image_width - inner_patch_buffer) % stride < inner_patch_buffer
        if possible_h and possible_w:
            possible_strides.append(stride)
    return possible_strides


class GridFullPatchedSubVolumeDataset(SubVolumeDataset):
    def __init__(
            self,
            image_path: Path,
            patch_size: int,
            z_start: int,
            z_size: int,
            load_labels: bool,
            transform: Optional[Callable] = None,
            stride: int = -1,
            num_dims: int = 5,
            loaded_papyrus: Optional[Dict] = None,
            compute_stride_start: int = 2,
            compute_stride_end: int = 32,
            compute_stride_step: int = 2,
            load_patch_size: Optional[int] = None,
            load_z_start: Optional[int] = None,
            load_z_size: Optional[int] = None,
            train: bool = True,
            inner_patch_buffer: int = -1,
    ):

        super().__init__(
            image_path=image_path,
            patch_size=patch_size,
            z_start=z_start,
            z_size=z_size,
            load_labels=load_labels,
            num_dims=num_dims,
            loaded_papyrus=loaded_papyrus,
            transform=transform,
            load_patch_size=load_patch_size,
            load_z_start=load_z_start,
            load_z_size=load_z_size,
            train=train)

        if stride == -1:
            stride = compute_possible_strides(
                image_height=self.image_height + self.padding,
                image_width=self.image_width + self.padding,
                stride_start=compute_stride_start,
                stride_end=compute_stride_end,
                stride_step=compute_stride_step)[-1]
            logger.debug(f'Stride not provided. Computed stride: {stride}')

        self.stride = stride

        assert self.stride % 2 == 0, 'Stride must be even'

        if inner_patch_buffer == -1:
            self.inner_patch_buffer = self.stride // 2
        else:
            self.inner_patch_buffer = inner_patch_buffer

        height_coordinates = [i for i in range(self.inner_patch_buffer, self.image_height + self.buffer, self.stride)]
        width_coordinates = [i for i in range(self.inner_patch_buffer, self.image_width + self.buffer, self.stride)]

        width_v, height_v = np.meshgrid(width_coordinates, height_coordinates, indexing='ij')
        grid_pixels = np.stack([height_v, width_v], axis=2).reshape(
            (len(height_coordinates) * len(width_coordinates), 2))
        grid_pixels = grid_pixels.astype(np.uint16)

        # Check if each inner patch has pixels inside the mask
        valid_pixels = []
        for pixel in grid_pixels:
            y, x = pixel + self.padding
            slice_y = slice(y - self.inner_patch_buffer, y + self.inner_patch_buffer, 1)
            slice_x = slice(x - self.inner_patch_buffer, x + self.inner_patch_buffer, 1)
            if (self.mask[slice_y, slice_x]).sum() != 0:
                valid_pixels.append(pixel)

        self.valid_pixels = np.array(valid_pixels)

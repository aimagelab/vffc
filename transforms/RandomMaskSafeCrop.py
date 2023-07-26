from typing import Dict, Any, Tuple
import numpy as np
import math
import torch.utils.data
import torch
import albumentations as A
from albumentations.augmentations.crops import crop
from albumentations import DualTransform
import random

from albumentations.core.transforms_interface import BoxInternalType, KeypointInternalType


class RandomMaskSafeCrop(DualTransform):

    def __init__(
        self,
        patch_size: int,
        always_apply: bool = True,
        p: float = 0.5,
    ):
        super(RandomMaskSafeCrop, self).__init__(always_apply, p)

        self.patch_size = patch_size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img[params['y_min']:params['y_max'], params['x_min']:params['x_max'], :]
        return img

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """used to generate parameters based on some targets. If your transform doesn't depend on any target,
        only it's own arguments, you can use get_params. These functions are used to produce params once per call,
        this is useful when you are producing some random or heavy params. """

        indices = np.argwhere(params['mask'] != 2)
        if len(indices) == 0:
            raise ValueError("Mask is empty")

        y0, y1 = np.min(indices[:, 0]), np.max(indices[:, 0])
        x0, x1 = np.min(indices[:, 1]), np.max(indices[:, 1])

        y_capped = False
        if y1 - y0 < self.patch_size:
            y_capped = True
            if y0 + self.patch_size // 2 <= params['mask'].shape[0]:
                y_min = max(0, y0 - self.patch_size // 2)
            else:
                y_min = max(0, y0 - self.patch_size)
        x_capped = False
        if x1 - x0 < self.patch_size:
            x_capped = True
            if x0 + self.patch_size // 2 <= params['mask'].shape[1]:
                x_min = max(0, x0 - self.patch_size // 2)
            else:
                x_min = max(0, x0 - self.patch_size)

        if random.random() > self.p:
            # Take the upper, leftmost patch
            y_min = y0 if not y_capped else y_min
            x_min = x0 if not x_capped else x_min
        else:
            y_min = random.randint(y0, y1 - self.patch_size) if not y_capped else y_min
            x_min = random.randint(x0, x1 - self.patch_size) if not x_capped else x_min

        y_max = y_min + self.patch_size
        x_max = x_min + self.patch_size
        return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        pass

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        pass

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        pass

    def apply_to_mask(self, mask, **params):
        return mask[params['y_min']:params['y_max'], params['x_min']:params['x_max']]


    @property
    def targets_as_params(self):
        """if you want to use some targets (arguments that you pass when call the augmentation pipeline) to produce
                some augmentation parameters on aug call, you need to list all of them here. When the transform is called,
                they will be provided in get_params_dependent_on_targets. For example: image, mask, bboxes, keypoints - are
                standard names for our targets. """
        return ["image", "mask"]

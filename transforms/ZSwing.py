import random
from typing import Any, Mapping, Tuple, Union

import numpy as np

from albumentations.core.transforms_interface import ImageOnlyTransform


class ZSwing(ImageOnlyTransform):
    def __init__(
        self,
        z_size: int,
        always_apply: bool = True,
        p: float = 0.5,
    ):
        super(ZSwing, self).__init__(always_apply, p)

        self.z_size = z_size

    def apply(self, img: np.ndarray, channels_to_drop: Tuple[int, ...] = (0,), **params) -> np.ndarray:
        img = img[:, :, params['z_min']:params['z_max']]
        return img

    def get_params_dependent_on_targets(self, params: Mapping[str, Any]):
        img = params["image"]

        load_z_size = img.shape[-1]

        if len(img.shape) == 2 or load_z_size == 1:
            raise NotImplementedError("Images has one depth channel. ZSwing is not defined.")

        if random.random() > self.p:
            # Take middle layers
            z_mid = load_z_size // 2
            z_min = z_mid - self.z_size // 2
        else:
            z_min = random.randint(0, load_z_size - self.z_size)

        z_max = z_min + self.z_size

        return {"z_min": z_min, "z_max": z_max}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "z_size",

    @property
    def targets_as_params(self):
        return ["image"]
# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2019-2023, Ross Wightman.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
###############################################################
# Code is based on the PyTorch Image Models (timm) library
# (https://github.com/huggingface/pytorch-image-models)
###############################################################

import math
import random
import warnings

import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


class ToNumpy:
    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


_torch_interpolation_to_str = {
    InterpolationMode.NEAREST: "nearest",
    InterpolationMode.BILINEAR: "bilinear",
    InterpolationMode.BICUBIC: "bicubic",
    InterpolationMode.BOX: "box",
    InterpolationMode.HAMMING: "hamming",
    InterpolationMode.LANCZOS: "lanczos",
}
_str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}


def str_to_interp_mode(mode_str):
    return _str_to_torch_interpolation[mode_str]


def _interp_mode_to_str(mode):
    return _torch_interpolation_to_str[mode]


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
    ----
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation="bilinear"):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)", stacklevel=2)

        self.interpolation = str_to_interp_mode(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
        ----
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
        -------
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.

        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
        ----
            img (PIL Image): Image to be cropped and resized.

        Returns
        -------
            PIL Image: Randomly cropped and resized image.

        """
        return F.resized_crop(img, *self.get_params(img, self.scale, self.ratio), self.size, self.interpolation)

    def __repr__(self):
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={_interp_mode_to_str(self.interpolation)})"
        return format_string


class ResizeKeepRatio:
    """Resize and Keep Ratio."""

    def __init__(
        self,
        size,
        longest=0.0,
        interpolation="bilinear",
        fill=0,
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = str_to_interp_mode(interpolation)
        self.longest = float(longest)
        self.fill = fill

    @staticmethod
    def get_params(img, target_size, longest):
        """Get parameters.

        Args:
        ----
            img (PIL Image): Image to be cropped.
            target_size (Tuple[int, int]): Size of output
        Returns:
            tuple: params (h, w) and (l, r, t, b) to be passed to ``resize`` and ``pad`` respectively

        """
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1.0 - longest)
        return [round(x / ratio) for x in source_size]

    def __call__(self, img):
        """
        Args:
        ----
            img (PIL Image): Image to be cropped and resized.

        Returns
        -------
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size

        """
        return F.resize(img, self.get_params(img, self.size, self.longest), self.interpolation)

    def __repr__(self):
        interpolate_str = _interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", interpolation={interpolate_str})"
        format_string += f", longest={self.longest:.3f})"
        return format_string

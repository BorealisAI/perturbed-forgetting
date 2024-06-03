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

from torchvision import transforms

from .constants import DEFAULT_CROP_PCT
from .transforms import RandomResizedCropAndInterpolation, ResizeKeepRatio, ToNumpy, str_to_interp_mode


def _transforms_imagenet_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    interpolation="bilinear",
):
    scale = tuple(scale or (0.05, 1.0))  # default imagenet scale range from big_vision
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    tfl = [RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        tfl += [transforms.RandomHorizontalFlip(p=hflip)]

    # prefetcher and collate will handle tensor conversion and norm
    tfl += [ToNumpy()]
    return transforms.Compose(tfl)


def _transforms_imagenet_eval(
    img_size=224,
    crop_pct=None,
    interpolation="bilinear",
):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        scale_size = tuple([math.floor(x / crop_pct) for x in img_size])
    else:
        scale_size = math.floor(img_size / crop_pct)
        scale_size = (scale_size, scale_size)

    # aspect ratio is preserved, crops center within image, no borders are added, image is lost
    if scale_size[0] == scale_size[1]:
        # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
        tfl = [transforms.Resize(scale_size[0], interpolation=str_to_interp_mode(interpolation))]
    else:
        # resize shortest edge to matching target dim for non-square target
        tfl = [ResizeKeepRatio(scale_size)]
    tfl += [transforms.CenterCrop(img_size)]

    # prefetcher and collate will handle tensor conversion and norm
    tfl += [ToNumpy()]
    return transforms.Compose(tfl)


def create_transform(
    input_size,
    is_training=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    interpolation="bilinear",
    crop_pct=None,
):
    img_size = input_size[-2:] if isinstance(input_size, (tuple, list)) else input_size

    if is_training:
        transform = _transforms_imagenet_train(
            img_size,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            interpolation=interpolation,
        )
    else:
        transform = _transforms_imagenet_eval(
            img_size,
            interpolation=interpolation,
            crop_pct=crop_pct,
        )

    return transform

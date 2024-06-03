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

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, Optional, Tuple

__all__ = ["PretrainedCfg", "DefaultCfg"]


@dataclass
class PretrainedCfg:
    file: Optional[str] = None  # filesystem path of the weights
    architecture: Optional[str] = None  # architecture variant can be set when not implicit
    tag: Optional[str] = None  # pretrained tag of source
    custom_load: bool = False  # use custom model specific model.load_pretrained() (ie for npz files)

    # input / data config
    input_size: Tuple[int, int, int] = (3, 224, 224)
    test_input_size: Optional[Tuple[int, int, int]] = None
    fixed_input_size: bool = False
    interpolation: str = "bicubic"
    crop_pct: float = 0.875
    test_crop_pct: Optional[float] = None
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # head / classifier config and meta-data
    num_classes: int = 1000
    classifier: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None or k == "classifier"}


@dataclass
class DefaultCfg:
    tags: Deque[str] = field(default_factory=deque)  # priority queue of tags (first is default)
    cfgs: Dict[str, PretrainedCfg] = field(default_factory=dict)  # pretrained cfgs by tag

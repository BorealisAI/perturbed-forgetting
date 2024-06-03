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

import os
from typing import Any, Dict, Optional, Union
from urllib.parse import urlsplit

from .pretrained import PretrainedCfg
from .registry import is_model, model_entrypoint, split_model_name_tag

__all__ = ["safe_model_name", "create_model"]


def _parse_model_name(model_name: str):
    return os.path.split(urlsplit(model_name).path)[-1]


def safe_model_name(model_name: str, remove_source: bool = True):
    # return a filename / path safe model name
    def _make_safe(name):
        return "".join(c if c.isalnum() else "_" for c in name).rstrip("_")

    if remove_source:
        model_name = _parse_model_name(model_name)
    return _make_safe(model_name)


def create_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Create a model.

    Lookup model's entrypoint function and pass relevant args to create a new model.

    Args:
    ----
        model_name: Name of model to instantiate.
        pretrained: If set to `True`, load pretrained ImageNet-1k weights.
        pretrained_cfg: Pass in an external pretrained_cfg for model.
        pretrained_cfg_overlay: Replace key-values in base pretrained_cfg with these.

    Keyword Args:
    ------------
        drop_rate (float): Classifier dropout rate for training.
        global_pool (str): Classifier global pooling type.

    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    model_name = _parse_model_name(model_name)
    model_name, pretrained_tag = split_model_name_tag(model_name)
    if pretrained_tag and not pretrained_cfg:
        # a valid pretrained_cfg argument takes priority over tag in model name
        pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise ValueError(f"Unknown model ({model_name})")

    create_fn = model_entrypoint(model_name)
    return create_fn(
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
        **kwargs,
    )

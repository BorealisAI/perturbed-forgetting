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

import dataclasses
import logging
from typing import Callable, Dict, Optional

from torch import nn

from .helpers import load_state_dict
from .pretrained import PretrainedCfg
from .registry import get_pretrained_cfg

_logger = logging.getLogger(__name__)

__all__ = ["build_model_with_cfg"]


def _load_pretrained(
    model: nn.Module,
    pretrained_cfg: Optional[Dict] = None,
    num_classes: int = 1000,
    in_chans: int = 3,
    strict: bool = True,
):
    """Load pretrained checkpoint.

    Args:
    ----
        model (nn.Module) : PyTorch model module
        pretrained_cfg (Optional[Dict]): configuration for pretrained weights / target dataset
        num_classes (int): num_classes for target model
        in_chans (int): in_chans for target model
        strict (bool): strict load of checkpoint

    """
    assert in_chans == 3
    pretrained_cfg = pretrained_cfg or getattr(model, "pretrained_cfg", None)
    if not pretrained_cfg:
        raise RuntimeError("Invalid pretrained config, cannot load weights. Use `pretrained=False` for random init.")

    pretrained_loc = pretrained_cfg.get("file", None)
    if not pretrained_loc:
        raise RuntimeError("Invalid pretrained config.")

    _logger.info(f"Loading pretrained weights from file ({pretrained_loc})")
    if pretrained_cfg.get("custom_load", False):
        model.load_pretrained(pretrained_loc)
        return
    state_dict = load_state_dict(pretrained_loc)

    classifiers = pretrained_cfg.get("classifier", None)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg["num_classes"]:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                state_dict.pop(classifier_name + ".weight", None)
                state_dict.pop(classifier_name + ".bias", None)
            strict = False

    load_result = model.load_state_dict(state_dict, strict=strict)
    if load_result.missing_keys:
        _logger.info(
            f'Missing keys ({", ".join(load_result.missing_keys)}) discovered while loading pretrained weights.'
            f" This is expected if model is being adapted.",
        )
    if load_result.unexpected_keys:
        _logger.warning(
            f'Unexpected keys ({", ".join(load_result.unexpected_keys)}) found while loading pretrained weights.'
            f" This may be expected if model is being adapted.",
        )


def _update_default_model_kwargs(pretrained_cfg, kwargs):
    """Update the default_cfg and kwargs before passing to model.

    Args:
    ----
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)

    """
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ("num_classes", "global_pool", "in_chans")
    if pretrained_cfg.get("fixed_input_size", False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ("img_size",)

    for n in default_kwarg_names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # pretrained_cfg has one input_size=(C, H ,W) entry
        if n == "img_size":
            input_size = pretrained_cfg.get("input_size", None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == "in_chans":
            input_size = pretrained_cfg.get("input_size", None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        elif n == "num_classes":
            default_val = pretrained_cfg.get(n, None)
            # if default is < 0, don't pass through to model
            if default_val is not None and default_val >= 0:
                kwargs.setdefault(n, pretrained_cfg[n])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])


def _resolve_pretrained_cfg(
    variant: str,
    pretrained_cfg=None,
    pretrained_cfg_overlay=None,
) -> PretrainedCfg:
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            # pretrained_cfg dict passed as arg, validate by converting to PretrainedCfg
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None

    # fallback to looking up pretrained cfg in model registry by variant identifier
    if not pretrained_cfg:
        if pretrained_tag:
            model_with_tag = f"{variant}.{pretrained_tag}"
        pretrained_cfg = get_pretrained_cfg(model_with_tag)

    if not pretrained_cfg:
        _logger.warning(
            f"No pretrained configuration specified for {model_with_tag} model. Using a default."
            f" Please add a config to the model pretrained_cfg registry or pass explicitly.",
        )
        pretrained_cfg = PretrainedCfg()  # instance with defaults

    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault("architecture", variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)

    return pretrained_cfg


def build_model_with_cfg(
    model_cls: Callable,
    variant: str,
    pretrained: bool,
    pretrained_cfg: Optional[Dict] = None,
    pretrained_cfg_overlay: Optional[Dict] = None,
    feature_cfg: Optional[Dict] = None,
    pretrained_strict: bool = True,
    **kwargs,
):
    """Build model with specified default_cfg.

    Args:
    ----
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        pretrained_cfg (dict): model's pretrained weight/task config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        pretrained_strict (bool): load pretrained weights strictly
        **kwargs: model args passed through to model __init__

    """
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = _resolve_pretrained_cfg(
        variant,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
    )

    # converting back to dict, PretrainedCfg use should be propagated further, but not into model
    pretrained_cfg = pretrained_cfg.to_dict()
    _update_default_model_kwargs(pretrained_cfg, kwargs)

    # Instantiate the model
    model = model_cls(**kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = getattr(model, "num_classes", kwargs.get("num_classes", 1000))
    if pretrained:
        _load_pretrained(
            model,
            pretrained_cfg=pretrained_cfg,
            num_classes=num_classes_pretrained,
            in_chans=kwargs.get("in_chans", 3),
            strict=pretrained_strict,
        )
    return model

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

import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .pretrained import DefaultCfg, PretrainedCfg

__all__ = [
    "split_model_name_tag",
    "generate_default_cfgs",
    "register_model",
    "is_model",
    "model_entrypoint",
    "get_pretrained_cfg",
]

_module_to_models: Dict[str, Set[str]] = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module: Dict[str, str] = {}  # mapping of model names to module names
_model_entrypoints: Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns
_model_has_pretrained: Set[str] = set()  # set of model names that have pretrained weight url present
_model_default_cfgs: Dict[str, PretrainedCfg] = {}  # central repo for model arch -> default cfg objects
_model_pretrained_cfgs: Dict[str, PretrainedCfg] = {}  # central repo for model arch.tag -> pretrained cfgs
_model_with_tags: Dict[str, List[str]] = defaultdict(list)  # shortcut to map each model arch to all model + tag names


def split_model_name_tag(model_name: str, no_tag: str = "") -> Tuple[str, str]:
    model_name, *tag_list = model_name.split(".", 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def generate_default_cfgs(cfgs: Dict[str, Union[Dict[str, Any], PretrainedCfg]]):
    out = defaultdict(DefaultCfg)

    for k, v in cfgs.items():
        if isinstance(v, dict):
            v = PretrainedCfg(**v)
        model, tag = split_model_name_tag(k)
        default_cfg = out[model]
        default_cfg.tags.append(tag)
        default_cfg.cfgs[tag] = v

    return out


def register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split(".")
    module_name = module_name_split[-1] if len(module_name_split) else ""

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    if model_name in _model_entrypoints:
        warnings.warn(
            f"Overwriting {model_name} in registry with {fn.__module__}.{model_name}. This is because the name being "
            "registered conflicts with an existing name. Please check if this is not expected.",
            stacklevel=2,
        )
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    if hasattr(mod, "default_cfgs") and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        default_cfg = mod.default_cfgs[model_name]

        for tag_idx, tag in enumerate(default_cfg.tags):
            is_default = tag_idx == 0
            pretrained_cfg = default_cfg.cfgs[tag]
            model_name_tag = f"{model_name}.{tag}" if tag else model_name
            replace_items = {"architecture": model_name, "tag": tag if tag else None}
            pretrained_cfg = replace(pretrained_cfg, **replace_items)

            if is_default:
                _model_pretrained_cfgs[model_name] = pretrained_cfg
                # add tagless entry if it's default and has weights
                _model_has_pretrained.add(model_name)

            _model_pretrained_cfgs[model_name_tag] = pretrained_cfg
            # add model w/ tag if tag is valid
            _model_has_pretrained.add(model_name_tag)
            _model_with_tags[model_name].append(model_name_tag)

        _model_default_cfgs[model_name] = default_cfg

    return fn


def _get_arch_name(model_name: str) -> str:
    return split_model_name_tag(model_name)[0]


def is_model(model_name: str) -> bool:
    """Check if a model name exists."""
    arch_name = _get_arch_name(model_name)
    return arch_name in _model_entrypoints


def model_entrypoint(model_name: str, module_filter: Optional[str] = None) -> Callable[..., Any]:
    """Fetch a model entrypoint for specified model name."""
    arch_name = _get_arch_name(model_name)
    if module_filter and arch_name not in _module_to_models.get(module_filter, {}):
        raise RuntimeError(f"Model ({model_name} not found in module {module_filter}.")
    return _model_entrypoints[arch_name]


def get_pretrained_cfg(model_name: str) -> PretrainedCfg:
    if model_name in _model_pretrained_cfgs:
        return deepcopy(_model_pretrained_cfgs[model_name])
    arch_name, tag = split_model_name_tag(model_name)
    if arch_name in _model_default_cfgs:
        # if model arch exists, but the tag is wrong, error out
        raise RuntimeError(f"Invalid pretrained tag ({tag}) for {arch_name}.")
    raise RuntimeError(f"Model architecture ({arch_name}) has no pretrained cfg registered.")

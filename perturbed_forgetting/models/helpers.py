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

import logging
import os
from typing import Any, Callable, Dict, Union

import torch
from torch import nn

_logger = logging.getLogger(__name__)

__all__ = ["named_apply", "load_state_dict", "load_checkpoint", "resume_checkpoint"]


def named_apply(
    fn: Callable,
    module: nn.Module,
    name="",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = f"{name}.{child_name}" if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def _clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_state_dict(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict_key = ""
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict_key = "state_dict"
            elif "model" in checkpoint:
                state_dict_key = "model"
        state_dict = _clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info(f"Loaded {state_dict_key} from checkpoint '{checkpoint_path}'")
        return state_dict

    _logger.error(f"No checkpoint found at '{checkpoint_path}'")
    raise FileNotFoundError


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
):
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, "load_pretrained"):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError("Model cannot load numpy checkpoint")
        return None

    state_dict = load_state_dict(checkpoint_path, device=device)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def resume_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    log_info: bool = True,
    not_found_ok: bool = False,
):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if log_info:
                _logger.info("Restoring model state from checkpoint...")
            state_dict = _clean_state_dict(checkpoint["state_dict"])
            model.load_state_dict(state_dict)

            if optimizer is not None and "optimizer" in checkpoint:
                if log_info:
                    _logger.info("Restoring optimizer state from checkpoint...")
                optimizer.load_state_dict(checkpoint["optimizer"])

            if "epoch" in checkpoint:
                resume_epoch = checkpoint["epoch"] + 1  # start at the next epoch
                if log_info:
                    _logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info(f"Loaded checkpoint '{checkpoint_path}'")
    else:
        not_found_str = f"No checkpoint found at '{checkpoint_path}'"
        if not_found_ok:
            _logger.info(not_found_str)
        else:
            raise FileNotFoundError(not_found_str)
    return resume_epoch

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

from typing import Optional

from torch import nn, optim

from .randsam import RandSAM
from .sam import SAM, RhoGenerator
from .sgdw import SGDW


def _param_groups_weight_decay(model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def optimizer_kwargs(cfg):
    """cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = {
        "opt": cfg.opt,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "momentum": cfg.momentum,
    }
    if getattr(cfg, "opt_eps", None) is not None:
        kwargs["eps"] = cfg.opt_eps
    if getattr(cfg, "opt_betas", None) is not None:
        kwargs["betas"] = cfg.opt_betas
    if getattr(cfg, "opt_args", None) is not None:
        kwargs.update(cfg.opt_args)

    opt_lower = cfg.opt.lower()
    opt_split = opt_lower.split("_")
    if len(opt_split) > 1 and "sam" in opt_split[0]:
        kwargs["sam_args"] = {
            "gsam_alpha": cfg.gsam_alpha,
            "adaptive": cfg.adaptive_sam,
            "asam_before_norm": cfg.asam_before_norm,
            "asam_after_norm": cfg.asam_after_norm,
            "backup_normalized": cfg.backup_normalized,
            "rho_generator": RhoGenerator(
                rho_policy=cfg.rho_policy,
                rho_max=cfg.rho,
                rho_min=cfg.min_rho,
                lr_max=cfg.lr,
                lr_min=cfg.min_lr,
            ),
        }

    return kwargs


def create_optimizer(
    model_or_params,
    opt: str = "sgd",
    lr: Optional[float] = None,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    filter_bias_and_bn: bool = True,
    **kwargs,
):
    """Create an optimizer.

    Args:
    ----
        model_or_params (nn.Module): model containing parameters to optimize
        opt: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
    -------
        Optimizer

    """
    if isinstance(model_or_params, nn.Module):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        no_weight_decay = {}
        if hasattr(model_or_params, "no_weight_decay"):
            no_weight_decay = model_or_params.no_weight_decay()

        if weight_decay and filter_bias_and_bn:
            parameters = _param_groups_weight_decay(model_or_params, weight_decay, no_weight_decay)
            weight_decay = 0.0
        else:
            parameters = model_or_params.parameters()
    else:
        # iterable of parameters or param groups passed in
        parameters = model_or_params

    opt_lower = opt.lower()
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    sam_args = kwargs.pop("sam_args", None)
    opt_args = dict(weight_decay=weight_decay, **kwargs)

    if lr is not None:
        opt_args.setdefault("lr", lr)

    # Base optimizers
    if opt_lower == "sgd":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(parameters, momentum=momentum, **opt_args)
    elif opt_lower == "sgdw":
        opt_args.pop("eps", None)
        optimizer = SGDW(parameters, momentum=momentum, **opt_args)
    elif opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        raise ValueError(f"Invalid optimizer {opt_lower}")

    # SAM
    if len(opt_split) > 1:
        if opt_split[0] == "sam":
            optimizer = SAM(optimizer, **sam_args)
        elif opt_split[0] == "randsam":
            optimizer = RandSAM(optimizer, **sam_args)
        else:
            raise ValueError(f"Invalid optimizer {opt_split[0]}")

    return optimizer

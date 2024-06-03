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

from torch.optim import Optimizer

from .linear_lr import LinearLRScheduler


def scheduler_kwargs(cfg):
    """cfg/argparse to kwargs helper
    Convert scheduler args in argparse args or cfg (.dot) like object to keyword args.
    """
    return {
        "sched": cfg.sched,
        "num_epochs": getattr(cfg, "epochs", 100),
        "warmup_epochs": getattr(cfg, "warmup_epochs", 5),
        "min_lr": getattr(cfg, "min_lr", 0.0),
        "warmup_lr": getattr(cfg, "warmup_lr", 1e-5),
    }


def create_scheduler(
    optimizer: Optimizer,
    sched: str = "linear",
    num_epochs: int = 300,
    min_lr: float = 0,
    warmup_lr: float = 1e-5,
    warmup_epochs: int = 0,
    updates_per_epoch: int = 0,
):
    t_initial = num_epochs
    warmup_t = warmup_epochs

    assert updates_per_epoch > 0, "updates_per_epoch must be set to number of dataloader batches"
    t_initial = t_initial * updates_per_epoch
    warmup_t = warmup_t * updates_per_epoch

    if sched == "linear":
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_t,
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched}")

    return lr_scheduler, num_epochs

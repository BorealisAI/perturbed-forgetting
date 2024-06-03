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

import torch

from .scheduler import Scheduler


class LinearLRScheduler(Scheduler):
    """Linear scheduler with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min: float = 0.0,
        warmup_t=0,
        warmup_lr_init=0,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_values(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            t_i = max(1, self.t_initial - self.warmup_t)
            lrs = [self.lr_min + (lr_max - self.lr_min) * (1 - (t / t_i)) for lr_max in self.base_values]

        return lrs

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

from typing import List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["SGDW"]


class SGDW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        *,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
        ----
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)
            _sgdw(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                maximize=group["maximize"],
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def _sgdw(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    maximize: bool,
):
    """Functional API that performs SGDW algorithm computation."""
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]
        param.mul_(1.0 - lr * weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            d_p = buf

        param.add_(d_p, alpha=-lr)

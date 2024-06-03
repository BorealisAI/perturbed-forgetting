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

import torch
import torch.nn.functional as F
from torch import nn


class BinaryCrossEntropy(nn.Module):

    def __init__(
        self,
        smoothing=0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert 0.0 <= smoothing <= 1.0
        self.smoothing = smoothing
        self.reductions = reduction.split("_")
        assert 1 <= len(self.reductions) <= 2
        assert all(r in ("mean", "sum") for r in self.reductions)
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        num_classes = x.shape[-1]
        if target.shape != x.shape:
            target = F.one_hot(target, num_classes=num_classes).to(x)
            if self.smoothing:
                target = ((1 - self.smoothing) * target) + (self.smoothing / num_classes)

        reduction = "sum" if len(self.reductions) > 1 else self.reductions[0]
        loss = F.binary_cross_entropy_with_logits(
            x,
            target,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction=reduction,
        )
        if len(self.reductions) > 1:
            denominator = 1.0
            if self.reductions[0] == "mean":
                denominator *= num_classes
            if self.reductions[1] == "mean":
                denominator *= x.shape[0]
            loss = loss / denominator
        return loss

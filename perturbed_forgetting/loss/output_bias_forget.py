# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
###############################################################

import torch
import torch.nn.functional as F
from torch import nn


class OutputBiasForget(nn.Module):
    """Output-Bias Forgetting perturbation function.

    SAM perturbations should maximize this loss. Doing so maximizes:
        OBF(pred, target) = (1 - alpha(pred, target)) * CE(pred, target) - CE(pred, uniform),
    where:
        alpha(pred, target) = gamma * max(0, (1 - lambda/pred[target]) / (1 - lambda)).

    NOTE: This implementation directly uses the gradient form, so the actual loss value is not very meaningful.

    Args:
    ----
        C (float): 1/lambda. For lambda=0, set C as negative, otherwise C > 1.
        gamma (float): Scaling for alpha in [0, 1]. Default: 1.
        sigmoid (bool): Whether to use sigmoid (default) or softmax for computing predictions.

    Returns:
    -------
        torch.Tensor (float): Loss to backprop with before taking a perturbing maximization step.

    """

    def __init__(self, C: float, gamma: float = 1, sigmoid: bool = True):
        super().__init__()
        assert not 0.0 <= C <= 1.0
        assert 0.0 <= gamma <= 1.0
        self.C = C
        self.gamma = gamma
        self.sigmoid = sigmoid

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        detached_x = x.detach()
        target_oh = F.one_hot(target, num_classes=detached_x.shape[-1]).to(detached_x)
        unif_target = torch.ones_like(detached_x) / detached_x.shape[-1]
        pred_ng = torch.sigmoid(detached_x) if self.sigmoid else F.softmax(detached_x, dim=-1)

        if self.C > 0:
            alpha = pred_ng[torch.arange(pred_ng.shape[0]), target, None]
            alpha = (self.C - (1 / alpha.clamp(min=1 / self.C))) / (self.C - 1)
        else:
            alpha = 1.0
        alpha = alpha * self.gamma

        perturb_loss = ((unif_target - ((alpha * pred_ng) + ((1 - alpha) * target_oh))) * x).sum(dim=-1).mean()
        return perturb_loss

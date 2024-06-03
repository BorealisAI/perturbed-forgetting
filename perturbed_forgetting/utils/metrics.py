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

import torch.nn.functional as F


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    maxk = min(max(topk), output.shape[-1])
    _, pred = output.topk(maxk, -1, True, True)
    pred = pred.permute(pred.dim() - 1, *range(pred.dim() - 1))
    correct = pred == target.unsqueeze(0)
    return [100.0 * correct[: min(k, maxk)].sum(0).float().mean() for k in topk]


def accuracy_multilabel(output, target, topk=(1,)):
    """Computes the multilabel accuracy over the k top predictions for the specified values of k.
    Returns whether each sample had a correct prediction for each of the topk values.
    """
    valid = target.sum(-1) > 0
    output, target = output[valid], target[valid]
    maxk = min(max(topk), output.shape[1])
    _, pred = output.topk(maxk, 1, True, True)
    pred = F.one_hot(pred, num_classes=target.shape[-1])
    correct = (pred * target.unsqueeze(-2)).sum(-1)
    correct = correct.cumsum(dim=-1).clamp(max=1).to(output)
    return [100.0 * correct[:, k - 1] for k in topk]

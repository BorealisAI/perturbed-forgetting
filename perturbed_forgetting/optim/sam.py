# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2022, Juntang Zhuang.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
###############################################################
# Code includes parts of https://github.com/juntang-zhuang/GSAM
###############################################################

import contextlib
from typing import List

import torch
from torch import distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["SAM", "RhoGenerator"]


def _maybe_no_sync(model):
    return model.no_sync() if dist.is_initialized() else contextlib.nullcontext()


@torch.jit.script
def _foreach_copy_(tensors: List[torch.Tensor], copy_tensors: List[torch.Tensor]) -> None:
    for i in range(len(tensors)):
        tensors[i].copy_(copy_tensors[i])


@torch.jit.script
def _foreach_reduced_norm(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(tensors)))


@torch.jit.script
def _foreach_normalize_(tensors: List[torch.Tensor], eps: float) -> None:
    torch._foreach_div_(tensors, _foreach_reduced_norm(tensors) + eps)


@torch.jit.script
def _foreach_reduced_dot(tensors1: List[torch.Tensor], tensors2: List[torch.Tensor]) -> torch.Tensor:
    t0 = tensors1[0]
    result = torch.zeros((), dtype=t0.dtype, device=t0.device)
    for i in range(len(tensors1)):
        result.add_(torch.dot(torch.flatten(tensors1[i]), torch.flatten(tensors2[i])))
    return result


class RhoGenerator:
    """Rho (perturbation size) generator for SAM."""

    def __init__(self, rho_policy, rho_max, rho_min=None, lr_max=None, lr_min=None):
        assert rho_policy in ("constant", "lr_prop")
        self.rho_policy = rho_policy
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.lr_max = lr_max
        self.lr_min = lr_min
        if rho_policy != "constant":
            self.rho_diff = self.rho_max - self.rho_min
            self.lr_diff = self.lr_max - self.lr_min
        self.lr_scheduler = None

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def get_rho(self, lr):
        if self.rho_policy == "constant":
            rho = self.rho_max
        else:  # lr_prop
            rho = max(0, ((lr - self.lr_min) / self.lr_diff) * self.rho_diff + self.rho_min)
        return rho


class SAM(torch.optim.Optimizer):
    """Optimizer with support for:
        - Sharpness-Aware Minimization (SAM): https://arxiv.org/abs/2010.01412
        - Surrogate Gap Guided SAM (GSAM): https://arxiv.org/abs/2203.08065
        - Adaptive SAM (ASAM): https://arxiv.org/abs/2102.11600

    Based on https://github.com/juntang-zhuang/GSAM and modified to:
        - Use PyTorch's `foreach` API for faster operations on lists of tensors
        - Support custom perturbation functions
        - Support gradient accumulation
        - Support m-sharpness
        - Configure if gradients normalized before backing up for GSAM decomposition
        - Configure adaptation before and after normalization for ASAM
        - Support separated rho generators
        - Other performance and readability optimizations
    """

    is_sam = True

    def __init__(
        self,
        base_optimizer,
        rho_generator,
        gsam_alpha,
        adaptive=False,
        asam_before_norm=False,
        asam_after_norm=False,
        backup_normalized=False,
        eps=1e-12,
    ):
        super().__init__(base_optimizer.param_groups, base_optimizer.defaults)
        if adaptive and not (asam_before_norm or asam_after_norm):
            raise ValueError("ASAM requires at least one of --asam-before-norm or --asam-after-norm.")
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.rho_generator = rho_generator
        self.alpha = gsam_alpha
        self.adaptive = adaptive
        self.asam_before_norm = asam_before_norm
        self.asam_after_norm = asam_after_norm
        self.backup_normalized = backup_normalized
        self.eps = eps
        self._reset()

    def _reset(self, init_params=True, init_old_grads=True):
        self.param_refs = []
        if init_params:
            self.params = []
            self.old_params = []
        if self.adaptive:
            self.abs_adaptive_params = []
        if self.alpha > 0 and init_old_grads:
            self.old_grads = []
            self.norm_grads = []
        self.accum_grads = []
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                self.param_refs.append(p)
                if init_params:
                    self.params.append(p.data)
                    self.old_params.append(torch.zeros_like(p.data))
                if self.adaptive:
                    self.abs_adaptive_params.append(torch.zeros_like(p.data))
                if self.alpha > 0 and init_old_grads:
                    self.old_grads.append(torch.zeros_like(p.data))
                    self.norm_grads.append(torch.zeros_like(p.data))
                self.accum_grads.append(torch.zeros_like(p.data))
        self.grads = None
        self.bn_modules = None

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
        self._reset()

    def _populate_grads(self):
        self.grads = [p.grad.data for p in self.param_refs]

    def _perturb_weights(self, delta):
        # NOTE right now we're assuming all groups have the same learning rate
        self.rho = self.rho_generator.get_rho(self.base_optimizer.param_groups[0]["lr"])
        torch._foreach_sub_(self.params, delta, alpha=self.rho)

    def _gradient_decompose(self, old_grads, alpha):
        # calculate projection norm
        _foreach_copy_(self.norm_grads, self.grads)
        _foreach_normalize_(self.norm_grads, self.eps)
        projection_norm = _foreach_reduced_dot(old_grads, self.norm_grads)

        # gradient decomposition (modifies old and new grads inplace)
        torch._foreach_sub_(old_grads, self.norm_grads, alpha=projection_norm)
        # NOTE Unlike the original GSAM, we are adding here instead of subtracting
        #      because the signs are flipped for the original gradients.
        torch._foreach_add_(self.grads, old_grads, alpha=alpha)

    def _populate_bn_modules(self, model):
        self.bn_modules = []
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                self.bn_modules.append(module)
        return self.bn_modules

    def _disable_running_stats(self, model):
        bn_modules = self.bn_modules if self.bn_modules is not None else self._populate_bn_modules(model)
        for module in bn_modules:
            module.backup_momentum = module.momentum
            module.momentum = 0

    def _enable_running_stats(self, model):
        bn_modules = self.bn_modules if self.bn_modules is not None else self._populate_bn_modules(model)
        for module in bn_modules:
            if hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

    @torch.no_grad()
    def backward(
        self,
        model,
        target,
        forward_logits_fn,
        loss_fn,
        perturb_loss_fn,
        sharpness_m,
        accum_steps=1,
        has_grad=False,
        need_update=True,
    ):
        accum_steps = accum_steps * (target.shape[0] // sharpness_m)

        for start_idx in range(0, target.shape[0], sharpness_m):
            # if has_grad is True, then some gradients are already computed that we need to back up
            if has_grad:
                self._populate_grads()
                _foreach_copy_(self.accum_grads, self.grads)
                self.base_optimizer.zero_grad()

            with _maybe_no_sync(model):
                # backup weights
                _foreach_copy_(self.old_params, self.params)
                if self.adaptive:
                    # get abs weights
                    _foreach_copy_(self.abs_adaptive_params, self.params)
                    torch._foreach_abs_(self.abs_adaptive_params)

                current_target = target[start_idx : start_idx + sharpness_m]

                # get gradient
                with torch.enable_grad():
                    logits = forward_logits_fn(start_idx, sharpness_m)
                    # negating for maximization
                    perturb_loss = -perturb_loss_fn(logits, current_target)
                    perturb_loss.backward()
                self._populate_grads()
                if self.adaptive and self.asam_before_norm:
                    torch._foreach_mul_(self.grads, self.abs_adaptive_params)

                # disable running stats for second pass
                self._disable_running_stats(model)

                # perturb weights
                if not self.backup_normalized and self.alpha > 0:
                    _foreach_copy_(self.old_grads, self.grads)
                _foreach_normalize_(self.grads, self.eps)
                if self.adaptive and self.asam_after_norm:
                    torch._foreach_mul_(self.grads, self.abs_adaptive_params)
                if self.backup_normalized and self.alpha > 0:
                    _foreach_copy_(self.old_grads, self.grads)
                self._perturb_weights(self.grads)

                # get gradient at perturbed weights
                self.base_optimizer.zero_grad()
                with torch.enable_grad():
                    logits = forward_logits_fn(start_idx, sharpness_m)
                    loss = loss_fn(logits, current_target)
                    loss.backward()
                self._populate_grads()

                # decompose and get new update direction (GSAM)
                if self.alpha > 0:
                    self._gradient_decompose(self.old_grads, self.alpha)

                # restore original params
                _foreach_copy_(self.params, self.old_params)

                # enable running stats
                self._enable_running_stats(model)

            # downscale for gradient accumulation
            if accum_steps > 1:
                torch._foreach_div_(self.grads, accum_steps)

                # restore accumulated gradients
                if has_grad:
                    torch._foreach_add_(self.grads, self.accum_grads)

            # we have grads now for the next internal batch
            has_grad = True

        # synchronize final gradients across workers
        if need_update and dist.is_initialized():
            for p in self.param_refs:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        return loss, perturb_loss

    @torch.no_grad()
    def step(self, closure=None):
        return self.base_optimizer.step(closure)

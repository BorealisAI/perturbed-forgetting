# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
###############################################################

from typing import List

import torch
from torch import distributed as dist

from .sam import SAM, _foreach_copy_, _foreach_normalize_, _maybe_no_sync


@torch.jit.script
def _foreach_randn_like(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.randn_like(t) for t in tensors]


class RandSAM(SAM):
    """SAM that perturbs weights with Gaussian noise."""

    def _populate_random_grads(self):
        self.grads = _foreach_randn_like(self.params)

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
                # update running stats
                logits = forward_logits_fn(start_idx, sharpness_m)
                perturb_loss = -perturb_loss_fn(logits, current_target)
                self._populate_random_grads()
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

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
import operator
import os

import torch

from .misc import get_state_dict, unwrap_model

_logger = logging.getLogger(__name__)


class CheckpointSaver:
    def __init__(
        self,
        model,
        optimizer,
        args=None,
        checkpoint_prefix="checkpoint",
        recovery_prefix="recovery",
        checkpoint_dir="",
        recovery_dir="",
        decreasing=False,
        max_history=10,
        unwrap_fn=unwrap_model,
    ):
        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.args = args

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ""
        self.last_recovery_file = ""

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = ".pth.tar"
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        self.unwrap_fn = unwrap_fn
        assert self.max_history >= 1

    def save_checkpoint(self, epoch, metric=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, "tmp" + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, "last" + self.extension)
        self._save(tmp_save_path, epoch, metric)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)  # required for Windows support
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if len(self.checkpoint_files) < self.max_history or metric is None or self.cmp(metric, worst_file[1]):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = "-".join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files,
                key=lambda x: x[1],
                reverse=not self.decreasing,
            )  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += f" {c}\n"
            _logger.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, "model_best" + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, epoch, metric=None):
        save_state = {
            "epoch": epoch,
            "arch": type(self.model).__name__.lower(),
            "state_dict": get_state_dict(self.model, self.unwrap_fn),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.args is not None:
            save_state["arch"] = self.args.model
            save_state["args"] = self.args
        if metric is not None:
            save_state["metric"] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        try:
            for d in to_delete:
                _logger.debug(f"Cleaning checkpoint: {d}")
                os.remove(d[0])
        except Exception:
            _logger.exception("Exception while deleting checkpoint")
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, epoch, batch_idx=0):
        assert epoch >= 0
        filename = "-".join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, epoch)
        if os.path.exists(self.last_recovery_file):
            try:
                _logger.debug(f"Cleaning recovery: {self.last_recovery_file}")
                os.remove(self.last_recovery_file)
            except Exception:
                _logger.exception(f"Exception while removing {self.last_recovery_file}")
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

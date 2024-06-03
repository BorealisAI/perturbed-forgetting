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

import contextlib
import logging
from functools import partial
from itertools import repeat

import numpy as np
import torch
import torch.utils.data

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .transforms_factory import create_transform

_logger = logging.getLogger(__name__)


def _fast_collate(batch):
    """A fast collation function optimized for uint8 images (np array or torch) and int64
    targets (labels). Supports multi-label datasets.
    """
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], np.ndarray):
        targets = [b[1] for b in batch]
        if isinstance(targets[0], np.ndarray):
            targets = np.array(targets)
        targets = torch.tensor(targets, dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
    elif isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
    else:
        raise RuntimeError("Unsupported batch item type.")
    return tensor, targets


def _adapt_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) != n:
        x_mean = np.mean(x).item()
        x = (x_mean,) * n
        _logger.warning(f"Pretrained mean/std different shape than model, using avg value {x}.")
    else:
        assert len(x) == n, "normalization stats must match image channels"
    return x


class _PrefetchLoader:

    def __init__(
        self,
        loader,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        channels=3,
        device="cuda",
        img_dtype=torch.float32,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        mean = _adapt_to_chs(mean, channels)
        std = _adapt_to_chs(std, channels)
        normalization_shape = (1, channels, 1, 1)

        self.loader = loader
        self.device = device
        self.img_dtype = img_dtype
        self.mean = torch.tensor([x * 255 for x in mean], device=device, dtype=img_dtype).view(normalization_shape)
        self.std = torch.tensor([x * 255 for x in std], device=device, dtype=img_dtype).view(normalization_shape)
        self.is_cuda = torch.cuda.is_available() and device.type == "cuda"

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = contextlib.nullcontext

        inputs, targets = None, None
        for next_inputs, next_targets in self.loader:

            with stream_context():
                next_inputs = next_inputs.to(device=self.device, non_blocking=True)
                next_targets = next_targets.to(device=self.device, non_blocking=True)
                next_inputs = next_inputs.to(self.img_dtype).sub_(self.mean).div_(self.std)

            if not first:
                yield inputs, targets
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            inputs = next_inputs
            targets = next_targets

        yield inputs, targets

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def _worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    np.random.seed(worker_info.seed % (2**32 - 1))


def create_loader(
    dataset,
    input_size,
    batch_size,
    is_training=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    interpolation="bilinear",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    num_workers=1,
    crop_pct=None,
    collate_fn=None,
    pin_memory=False,
    img_dtype=torch.float32,
    device="cuda",
    persistent_workers=True,
):
    if isinstance(device, str):
        device = torch.device(device)
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        interpolation=interpolation,
        crop_pct=crop_pct,
    )

    # give TFDS datasets early knowledge of num_workers so that sample estimates
    # are correct before worker processes are launched
    dataset.set_loader_cfg(num_workers=num_workers)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and is_training,
        num_workers=num_workers,
        collate_fn=collate_fn or _fast_collate,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=_worker_init,
        persistent_workers=persistent_workers,
    )

    return _PrefetchLoader(
        loader,
        mean=mean,
        std=std,
        channels=input_size[0],
        device=device,
        img_dtype=img_dtype,
    )

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

from .tfds_dataset import TfdsDataset


def create_tfds_dataset(
    name,
    root,
    split="validation",
    is_training=False,
    download=False,
    batch_size=None,
    seed=42,
    **kwargs,
):
    """Dataset factory method.

    Args:
    ----
        name: dataset name, empty is okay for folder based datasets
        root: root folder of dataset
        split: dataset split
        is_training: create dataset in train mode, which enables shuffling
        download: download dataset if not present
        batch_size: batch size hint
        seed: random seed
        **kwargs: other args to pass to dataset

    Returns:
    -------
        Dataset object

    """
    name = name.lower()
    assert name.startswith("tfds/")

    return TfdsDataset(
        root,
        name=name,
        split=split,
        is_training=is_training,
        download=download,
        batch_size=batch_size,
        seed=seed,
        **kwargs,
    )

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

import argparse
import ast
import collections.abc
import logging
import logging.handlers
import os
import random
from itertools import repeat

import numpy as np
import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


class ParseKwargs(argparse.Action):
    def __call__(self, _parser, namespace, values, _option_string=None):
        kw = {}
        for value in values:
            key, value = value.split("=")
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


class _FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt="%(levelname)s: %(message)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + "-" + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + "-" + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))

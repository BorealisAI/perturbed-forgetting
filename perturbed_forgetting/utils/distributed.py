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

import os

import torch
from torch import distributed as dist

from .misc import unwrap_model


def reduce_tensor(tensor, clone=True):
    rt = tensor.clone() if clone else tensor
    dist.all_reduce(rt, op=dist.ReduceOp.AVG)
    return rt


def distribute_bn(model, reduce=False):
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ("running_mean" in bn_name) or ("running_var" in bn_name):
            if reduce:
                # average bn stats across whole group
                dist.all_reduce(bn_buf, op=dist.ReduceOp.AVG)
            else:
                # broadcast bn stats from rank 0 to whole group
                dist.broadcast(bn_buf, 0)


def _is_global_primary(args):
    return args.rank == 0


def _is_local_primary(args):
    return args.local_rank == 0


def is_primary(args, local=False):
    return _is_local_primary(args) if local else _is_global_primary(args)


def _is_distributed_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def _world_info_from_env():
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    dist_backend = getattr(args, "dist_backend", "nccl")
    dist_url = getattr(args, "dist_url", "env://")
    if _is_distributed_env():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = _world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ["LOCAL_RANK"] = str(args.local_rank)
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            dist.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = _world_info_from_env()
            dist.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
            )
            args.world_size = dist.get_world_size()
            args.rank = dist.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        device = "cuda:%d" % args.local_rank if args.distributed else "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    args.device = device
    return torch.device(device)

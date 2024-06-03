#!/usr/bin/env python3
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
import contextlib
import copy
import json
import logging
import os
import time
from collections import OrderedDict
from datetime import datetime
from functools import partial

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from perturbed_forgetting import utils
from perturbed_forgetting.data import create_loader, create_tfds_dataset, resolve_data_config
from perturbed_forgetting.loss import BinaryCrossEntropy, OutputBiasForget
from perturbed_forgetting.models import create_model, resume_checkpoint, safe_model_name
from perturbed_forgetting.optim import create_optimizer, optimizer_kwargs
from perturbed_forgetting.scheduler import create_scheduler, scheduler_kwargs

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

_logger = logging.getLogger("train")


def _positive_or_none(dtype):
    def f(value):
        value = dtype(value)
        return value if value > 0 else None

    return f


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# Dataset parameters
group = parser.add_argument_group("Dataset parameters")
parser.add_argument("--data-dir", metavar="DIR", help="path to dataset (root dir)")
parser.add_argument(
    "--dataset",
    metavar="NAME",
    default="",
    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)',
)
group.add_argument("--train-split", metavar="NAME", default="train", help="dataset train split (default: train)")
group.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
group.add_argument(
    "--dataset-download",
    action="store_true",
    default=False,
    help="Allow download of dataset for torch/ and tfds/ datasets that support it.",
)

# Model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument(
    "--model",
    default="resnet50",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "resnet50")',
)
group.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
group.add_argument(
    "--pretrained-path",
    default=None,
    type=str,
    help="Load this checkpoint as if they were the pretrained weights (with adaptation).",
)
group.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint; 'auto' for auto-resume (default: none)",
)
group.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (Model default if None)",
)
group.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
group.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image size (default: None => model default)",
)
group.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
group.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
group.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of dataset",
)
group.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
group.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="Input batch size for training (default: 128)",
)
group.add_argument(
    "-vb",
    "--validation-batch-size",
    type=int,
    default=None,
    metavar="N",
    help="Validation batch size override (default: None)",
)
group.add_argument(
    "--grad-accum-steps",
    type=int,
    default=1,
    metavar="N",
    help="The number of steps to accumulate gradients (default: 1)",
)
group.add_argument("--model-kwargs", nargs="*", default={}, action=utils.ParseKwargs)
group.add_argument("--head-init-bias", default=None, type=float, help="Head initialization bias value")

# Optimizer parameters
group = parser.add_argument_group("Optimizer parameters")
group.add_argument("--opt", default="sgd", type=str, metavar="OPTIMIZER", help='Optimizer (default: "sgd")')
group.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
group.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
group.add_argument("--momentum", type=float, default=0.9, metavar="M", help="Optimizer momentum (default: 0.9)")
group.add_argument("--weight-decay", type=float, default=2e-5, help="weight decay (default: 2e-5)")
group.add_argument(
    "--clip-grad",
    type=_positive_or_none(float),
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
group.add_argument(
    "--snp-factor",
    type=float,
    default=0,
    help="shrink and perturb factor. 0=none, 1=full reset. (default: 0)",
)
group.add_argument("--sharpness-m", type=int, default=-1, help="Sharpness M (use batch size if -1)")
group.add_argument("--gsam-alpha", type=float, default=0, help="GSAM alpha. When alpha=0, GSAM is equivalent to SAM.")
group.add_argument("--adaptive-sam", action="store_true", default=False, help="Adaptive perturbations like ASAM.")
group.add_argument(
    "--asam-before-norm",
    action="store_true",
    default=False,
    help="Scale gradients before normalization for ASAM.",
)
group.add_argument(
    "--asam-after-norm",
    action="store_true",
    default=False,
    help="Scale gradients after normalization for ASAM.",
)
group.add_argument(
    "--backup-normalized",
    action="store_true",
    default=False,
    help="Backup normalized clean gradients in GSAM.",
)
group.add_argument(
    "--rho-policy",
    type=str,
    default="constant",
    choices=["constant", "lr_prop"],
    help="How rho should be set in *SAM",
)
group.add_argument("--rho", type=float, default=0.6, help="Value of rho (maximum when scheduling)")
group.add_argument("--min-rho", type=float, default=0.0, help="Minimum value of rho")
group.add_argument("--opt-kwargs", nargs="*", default={}, action=utils.ParseKwargs)
group.add_argument("--bce-loss", action="store_true", default=False, help="Enable BCE loss.")
group.add_argument("--no-bce-loss", dest="bce_loss", action="store_false")
group.add_argument(
    "--bce-reduction",
    type=str,
    default="sum_mean",
    help="Reduction to apply when using BCE. r1_r2 applies r1 across targets and r2 across batch",
)
group.add_argument(
    "--perturb-loss",
    type=str,
    default="maximize",
    choices=["maximize", "obf"],
    help="Loss for perturbation in *SAM",
)
group.add_argument(
    "--obf-C",
    type=float,
    default=1000,
    help="Reciprocal of OBF lambda; can be set of num classes (-1 for lambda=0)",
)
group.add_argument("--obf-gamma", type=float, default=1, help="Scale for OBF dynamic uniformity strength")

# Learning rate schedule parameters
group = parser.add_argument_group("Learning rate schedule parameters")
group.add_argument("--sched", type=str, default="linear", metavar="SCHEDULER", help="LR scheduler")
group.add_argument(
    "--lr",
    type=float,
    default=None,
    metavar="LR",
    help="learning rate, overrides lr-base if set (default: None)",
)
group.add_argument(
    "--lr-base",
    type=float,
    default=0.1,
    metavar="LR",
    help="base learning rate: lr = lr_base * global_batch_size / base_size",
)
group.add_argument(
    "--lr-base-size",
    type=int,
    default=256,
    metavar="DIV",
    help="base learning rate batch size (divisor, default: 256).",
)
group.add_argument(
    "--lr-base-scale",
    type=str,
    default="",
    metavar="SCALE",
    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)',
)
group.add_argument("--warmup-lr", type=float, default=1e-5, metavar="LR", help="warmup learning rate (default: 1e-5)")
group.add_argument(
    "--min-lr",
    type=float,
    default=0,
    metavar="LR",
    help="learning rate at the end of the last epoch (default: 0)",
)
group.add_argument("--epochs", type=int, default=300, metavar="N", help="number of epochs to train (default: 300)")
group.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
group.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)

# Augmentation & regularization parameters
group = parser.add_argument_group("Augmentation and regularization parameters")
group.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.05, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.05 1.0)",
)
group.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
group.add_argument("--hflip", type=float, default=0.5, help="Horizontal flip training aug probability")
group.add_argument("--smoothing", type=float, default=0.0, help="Label smoothing")
group.add_argument(
    "--train-interpolation",
    type=str,
    default="bilinear",
    help="Training interpolation (bilinear, bicubic)",
)
group.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)")

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group("Batch norm parameters", "Only works with gen_efficientnet based models currently.")
group.add_argument("--bn-momentum", type=float, default=None, help="BatchNorm momentum override (if not None)")
group.add_argument("--bn-eps", type=float, default=None, help="BatchNorm epsilon override (if not None)")
group.add_argument(
    "--dist-bn",
    type=str,
    default="reduce",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)

# Misc
group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")
group.add_argument(
    "--log-interval",
    type=int,
    default=50,
    metavar="N",
    help="how many batches to wait before logging training status",
)
group.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
group.add_argument(
    "--checkpoint-hist",
    type=int,
    default=10,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
group.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 4)",
)
group.add_argument(
    "--synchronize-step",
    action="store_true",
    default=False,
    help="torch.cuda.synchronize() end of each step",
)
group.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
group.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
group.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
group.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1"',
)
group.add_argument("--local_rank", default=0, type=int)
group.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)


def main():
    utils.setup_default_logging()
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}.",
        )
    else:
        _logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    utils.random_seed(args.seed, args.rank)

    factory_kwargs = {}
    if args.pretrained_path:
        factory_kwargs["pretrained_cfg_overlay"] = {
            "file": args.pretrained_path,
            "num_classes": -1,  # force head adaptation
        }

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=3,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        **factory_kwargs,
        **args.model_kwargs,
    )
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    if utils.is_primary(args):
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}",
        )

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # move model to GPU
    model.to(device=device)
    init_params = copy.deepcopy(model.state_dict()) if args.snp_factor > 0 else None

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = "sqrt" if any(o in on for o in ("ada", "lamb")) else "linear"
        if args.lr_base_scale == "sqrt":
            batch_ratio = batch_ratio**0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f"Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) "
                f"and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.",
            )

    optimizer = create_optimizer(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    if utils.is_primary(args):
        _logger.info(f"Optimizer:\n{optimizer}")

    # optionally resume from a checkpoint
    resume_epoch = None
    resume_auto = args.resume == "auto"
    if resume_auto:
        args.resume = os.path.join(args.output, args.experiment, "last.pth.tar")
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            log_info=utils.is_primary(args),
            not_found_ok=resume_auto,
        )

    # setup distributed training
    if args.distributed:
        if utils.is_primary(args):
            _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[device])

    # create the train and eval datasets
    dataset_train = create_tfds_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    dataset_eval = create_tfds_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        download=args.dataset_download,
        batch_size=args.validation_batch_size or args.batch_size,
    )

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if not train_interpolation:
        train_interpolation = data_config["interpolation"]
    loader_train = create_loader(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=True,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        device=device,
    )

    eval_workers = args.workers
    if args.distributed:
        # reduces validation padding issues when using TFDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=eval_workers,
        crop_pct=data_config["crop_pct"],
        pin_memory=args.pin_mem,
        device=device,
    )

    # print train and eval transformations
    if utils.is_primary(args):
        _logger.info(f"Train transforms:\n{loader_train.dataset.transform}")
        _logger.info(f"Eval transforms:\n{loader_eval.dataset.transform}")

    # setup loss function
    if args.bce_loss:
        train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, reduction=args.bce_reduction)
    else:
        train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    if getattr(optimizer, "is_sam", False):
        if args.perturb_loss == "maximize":
            perturb_loss_fn = train_loss_fn
        elif args.perturb_loss == "obf":
            perturb_loss_fn = OutputBiasForget(args.obf_C, gamma=args.obf_gamma)
    else:
        perturb_loss_fn = None
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                ],
            )
        output_dir = utils.get_outdir(args.output if args.output else "./output/train", exp_name)
        decreasing = eval_metric == "loss"
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )

    wandb_initialized = False
    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project="perturbed-forgetting", id=args.experiment, resume="allow", config=args)
            wandb_initialized = True
        else:
            raise RuntimeError("wandb logging was requested but wandb package was not found")
    args.log_wandb = wandb_initialized

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step_update(start_epoch * updates_per_epoch)
    if getattr(optimizer, "is_sam", False):
        optimizer.rho_generator.set_lr_scheduler(lr_scheduler)

    if utils.is_primary(args):
        _logger.info(f"Scheduled epochs: {num_epochs}. LR stepped per update.")

    results = [None] * start_epoch
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                perturb_loss_fn=perturb_loss_fn,
                init_params=init_params,
            )

            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.dist_bn == "reduce")

            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                num_updates=((epoch + 1) * updates_per_epoch),
                epoch=(epoch + 1),
            )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            results.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": eval_metrics,
                },
            )

    except KeyboardInterrupt:
        pass

    results = {"all": results}
    if best_metric is not None:
        results["best"] = results["all"][best_epoch]
        _logger.info(f"*** Best metric: {best_metric} (epoch {best_epoch})")
    print(f"--result\n{json.dumps(results, indent=4)}")


def _forward_backward(
    model,
    optimizer,
    loss_fn,
    inputs,
    targets,
    has_grad,
    accum_steps,
    need_update,
    has_no_sync,
    clip_grad,
    is_sam,
    sharpness_m,
    perturb_loss_fn,
):
    """Perform forward and backward passes, with weights update if needed."""

    def _forward_logits(model, inp, start_idx, num_samples):
        return model(inp[start_idx : start_idx + num_samples])

    forward_logits = partial(_forward_logits, model, inputs)

    fb_context = model.no_sync if has_no_sync and not need_update else contextlib.nullcontext
    with fb_context():
        if not is_sam:
            loss = perturb_loss = loss_fn(forward_logits(0, inputs.shape[0]), targets)
            (loss / accum_steps).backward()
        else:
            loss, perturb_loss = optimizer.backward(
                model,
                targets,
                forward_logits,
                loss_fn,
                perturb_loss_fn,
                sharpness_m,
                accum_steps=accum_steps,
                has_grad=has_grad,
                need_update=need_update,
            )
        if need_update:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
    return loss, perturb_loss


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    device="cuda",
    lr_scheduler=None,
    saver=None,
    perturb_loss_fn=None,
    init_params=None,
):
    if isinstance(device, str):
        device = torch.device(device)
    has_no_sync = hasattr(model, "no_sync")
    is_sam = getattr(optimizer, "is_sam", False)
    if is_sam:
        if perturb_loss_fn is None:
            raise RuntimeError("SAM requires perturb loss function.")
        sharpness_m = args.sharpness_m if args.sharpness_m > 0 else args.batch_size
        if sharpness_m > args.batch_size:
            raise RuntimeError("Sharpness M cannot be larger than local batch size.")
        if args.batch_size % sharpness_m != 0:
            raise RuntimeError("Local batch size must be divisible by Sharpness M.")

    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    perturb_losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % args.grad_accum_steps
    updates_per_epoch = (len(loader) + args.grad_accum_steps - 1) // args.grad_accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx == args.grad_accum_steps * 3:
            update_time_m.reset()
            data_time_m.reset()
            losses_m.reset()
            perturb_losses_m.reset()

        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % args.grad_accum_steps == 0
        update_idx = batch_idx // args.grad_accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(args.grad_accum_steps * (time.time() - data_start_time))

        if batch_idx % args.grad_accum_steps == 0 and args.snp_factor > 0:
            snp_factor = args.snp_factor * (optimizer.param_groups[0]["lr"] / lr_scheduler.base_values[0])
            for k, v in model.state_dict().items():
                if args.distributed:
                    k = k.split(".", 1)[1]
                v.data.copy_((snp_factor * init_params[k].data) + ((1 - snp_factor) * v.data))
                if args.distributed:
                    dist.broadcast(v, 0)

        loss, perturb_loss = _forward_backward(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            inputs=inputs,
            targets=targets,
            has_grad=(batch_idx % args.grad_accum_steps > 0),
            accum_steps=accum_steps,
            need_update=need_update,
            has_no_sync=has_no_sync,
            clip_grad=args.clip_grad,
            is_sam=is_sam,
            sharpness_m=sharpness_m if is_sam else None,
            perturb_loss_fn=perturb_loss_fn,
        )

        losses_m.update(loss.item(), inputs.size(0))
        perturb_losses_m.update(perturb_loss.item(), inputs.size(0))
        update_sample_count += inputs.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()

        if args.synchronize_step and device.type == "cuda":
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]

            if args.distributed:
                update_sample_count *= args.world_size

            if args.log_wandb:
                log_dict = {
                    "train/loss": losses_m.val,
                    "train/perturb_loss": perturb_losses_m.val,
                    "lr": lr,
                }
                wandb.log(log_dict, step=num_updates - 1)

            if utils.is_primary(args):
                _logger.info(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                    f"({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  "
                    f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                    f"Perturb loss: {perturb_losses_m.val:#.3g} ({perturb_losses_m.avg:#.3g})  "
                    f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                    f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  "
                    f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})",
                )

        if saver is not None and args.recovery_interval and ((update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()

    return OrderedDict([("loss", losses_m.avg), ("perturb_loss", perturb_losses_m.avg)])


def validate(
    model,
    loader,
    loss_fn,
    args,
    device="cuda",
    log_suffix="",
    num_updates=None,
    epoch=None,
):
    if isinstance(device, str):
        device = torch.device(device)
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    num_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            last_batch = batch_idx == num_batches - 1

            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = loss_fn(outputs, targets)
            acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data)
                acc1 = utils.reduce_tensor(acc1, clone=False)
                acc5 = utils.reduce_tensor(acc5, clone=False)
            else:
                reduced_loss = loss.data

            if device.type == "cuda":
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), outputs.size(0))
            top5_m.update(acc5.item(), outputs.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = "Test" + log_suffix
                _logger.info(
                    f"{log_name}: [{batch_idx:>4d}/{num_batches}]  "
                    f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  "
                    f"Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  "
                    f"Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})",
                )

    metrics = OrderedDict(
        [("loss" + log_suffix, losses_m.avg), ("top1" + log_suffix, top1_m.avg), ("top5" + log_suffix, top5_m.avg)],
    )
    if args.log_wandb and num_updates is not None:
        wandb_metrics = {"eval/" + k: v for k, v in metrics.items()}
        if epoch is not None:
            wandb_metrics["epoch"] = epoch
        wandb.log(wandb_metrics, step=num_updates)

    return metrics


if __name__ == "__main__":
    main()

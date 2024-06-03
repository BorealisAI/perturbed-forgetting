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
import json
import logging
import sys
import time
from collections import OrderedDict

import torch
import torch.nn.parallel
from pyhessian import hessian
from torch import nn

from perturbed_forgetting.data import create_loader, create_tfds_dataset, resolve_data_config
from perturbed_forgetting.loss import BinaryCrossEntropy
from perturbed_forgetting.models import create_model, is_model, load_checkpoint
from perturbed_forgetting.utils import AverageMeter, ParseKwargs, accuracy, accuracy_multilabel, setup_default_logging

_logger = logging.getLogger("validate")


parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("--data-dir", metavar="DIR", help="path to dataset (root dir)")
parser.add_argument(
    "--dataset",
    metavar="NAME",
    default="",
    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)',
)
parser.add_argument("--split", metavar="NAME", default="validation", help="dataset split (default: validation)")
parser.add_argument(
    "--dataset-download",
    action="store_true",
    default=False,
    help="Allow download of dataset for torch/ and tfds/ datasets that support it.",
)
parser.add_argument("--model", "-m", metavar="NAME", default="resnet50", help="model architecture (default: resnet50)")
parser.add_argument(
    "--validate-hessian",
    action="store_true",
    default=False,
    help="Evaluate the Hessian information instead of accuracy.",
)
parser.add_argument(
    "--sharpness-iters",
    type=int,
    default=100,
    help="[Hessian only] Number of iterations for power iteration",
)
parser.add_argument(
    "--sharpness-tol",
    type=float,
    default=1e-4,
    help="[Hessian only] Tolerance for finishing power iteration",
)
parser.add_argument("--smoothing", type=float, default=0.0, help="[Hessian only] Label smoothing for loss")
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="bilinear",
    help="[Hessian only] Training interpolation (bilinear, bicubic)",
)
parser.add_argument("--bce-loss", action="store_true", default=False, help="Enable BCE loss.")
parser.add_argument("--no-bce-loss", dest="bce_loss", action="store_false")
parser.add_argument(
    "--bce-reduction",
    type=str,
    default="sum_mean",
    help="Reduction to apply when using BCE. r1_r2 applies r1 across targets and r2 across batch",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N", help="mini-batch size (default: 256)")
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--use-train-size",
    action="store_true",
    default=False,
    help="force use of train input size, even when test size is specified in pretrained cfg",
)
parser.add_argument("--crop-pct", default=None, type=float, metavar="N", help="Input image center crop pct")
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument("--num-classes", type=int, default=None, help="Number classes in dataset")
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument("--log-freq", default=10, type=int, metavar="N", help="batch logging frequency (default: 10)")
parser.add_argument(
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)


def validate(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if not args.validate_hessian:
            torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    else:
        _logger.warning("No checkpoint specified. Evaluation will use random weights!")

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info(f"Model {args.model} created, param count: {param_count}")

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )

    model = model.to(device)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    if args.bce_loss:
        criterion = BinaryCrossEntropy(smoothing=args.smoothing, reduction=args.bce_reduction).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing).to(device)

    dataset = create_tfds_dataset(
        root=args.data_dir,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
    )

    crop_pct = data_config["crop_pct"]
    train_interpolation = args.train_interpolation or data_config["interpolation"]
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        interpolation=data_config["interpolation"] if not args.validate_hessian else train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        device=device,
    )
    _logger.info(f"Loader transforms:\n{loader.dataset.transform}")

    model.eval()

    if args.validate_hessian:
        _logger.info(f"Dataloader has {len(loader)} batches")
        hessian_comp = hessian(model, criterion, dataloader=loader)
        model.zero_grad()
        max_ev = hessian_comp.eigenvalues(maxIter=args.sharpness_iters, tol=args.sharpness_tol, top_n=1)[0][0]
        print("Dominant eigenvalue: ", max_ev)
        sys.exit(0)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        inputs = torch.randn((args.batch_size, *tuple(data_config["input_size"]))).to(device)
        model(inputs)

        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(loader):
            # compute output
            outputs = model(inputs)
            ce_targets = targets.argmax(dim=1) if targets.ndim > 1 else targets
            loss = criterion(outputs, ce_targets)
            losses.update(loss.item(), inputs.size(0))

            # measure accuracy and record loss
            if targets.ndim > 1:
                acc1, acc5 = accuracy_multilabel(outputs.detach(), targets, topk=(1, 5))
                top1.update(acc1.mean().item(), acc1.size(0))
                top5.update(acc5.mean().item(), acc5.size(0))
            else:
                acc1, acc5 = accuracy(outputs.detach(), targets, topk=(1, 5))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    f"Test: [{batch_idx:>4d}/{len(loader)}]  "
                    f"Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {inputs.size(0) / batch_time.avg:>7.2f}/s)  "
                    f"Loss: {losses.val:>7.4f} ({losses.avg:>6.4f})  "
                    f"Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
                    f"Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})",
                )

    top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4),
        top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4),
        top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config["input_size"][-1],
        crop_pct=crop_pct,
        interpolation=data_config["interpolation"],
    )

    _logger.info(
        f" * Acc@1 {results['top1']:.3f} ({results['top1_err']:.3f})"
        f" Acc@5 {results['top5']:.3f} ({results['top5_err']:.3f})",
    )
    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    if not is_model(args.model):
        raise ValueError(f"Unknown model: {args.model}")

    results = validate(args)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f"--result\n{json.dumps(results, indent=4)}")


if __name__ == "__main__":
    main()

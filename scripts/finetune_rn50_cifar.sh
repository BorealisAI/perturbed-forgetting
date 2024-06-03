#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
###############################################################

set -e

cd /path/to/perturbed_forgetting

export NCPU=${NCPU:-4}

export NUM_CLASSES=${NUM_CLASSES:-10}

export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-128}
export SHARPNESS_M=${SHARPNESS_M:-64}
export BATCH_SIZE=512

export OMP_NUM_THREADS=$((NCPU*2))
export TFDS_TP_SIZE=$((NCPU*2))
export TFDS_SHUFFLE_SIZE=8192
export TFDS_AUTOTUNE_BYTES=$((1024*1024*1024*2))
export NO_GCE_CHECK=true

LD_PRELOAD="/path/to/libtcmalloc.so.4:$LD_PRELOAD" \
    torchrun --nproc_per_node=1 train.py \
    --model resnet50 \
    --pretrained \
    --dataset "tfds/cifar$NUM_CLASSES" \
    --dataset-download \
    --num-classes $NUM_CLASSES \
    --train-split "train" \
    --val-split "test" \
    --data-dir /path/to/tensorflow_datasets \
    --pin-mem \
    --output /path/to/output \
    --experiment "$EXP_NAME" \
    --crop-pct 0.875 \
    --train-interpolation bilinear \
    --resume auto \
    --workers $NCPU \
    --batch-size $LOCAL_BATCH_SIZE \
    --sharpness-m $SHARPNESS_M \
    --grad-accum-steps $((BATCH_SIZE/LOCAL_BATCH_SIZE)) \
    --validation-batch-size 64 \
    --epochs 100 \
    --warmup-epochs 5 \
    --lr 0.01 \
    --min-lr 0 \
    --warmup-lr 0 \
    --weight-decay 0 \
    --momentum 0.9 \
    --clip-grad 1.0 \
    --gsam-alpha 0 \
    --min-rho 0.0 \
    --rho 0.1 \
    --obf-C $NUM_CLASSES \
    $@

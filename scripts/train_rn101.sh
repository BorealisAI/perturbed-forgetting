#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
###############################################################

set -e

cd /path/to/perturbed_forgetting

export NGPU=${NGPU:-1}
export NCPU=${NCPU:-4}

export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-256}
export SHARPNESS_M=${SHARPNESS_M:-64}
export BATCH_SIZE=4096

export OMP_NUM_THREADS=$((NCPU*2))
export TFDS_TP_SIZE=$((NCPU*2))
export TFDS_SHUFFLE_SIZE=8192
export TFDS_AUTOTUNE_BYTES=$((1024*1024*1024*2))
export NO_GCE_CHECK=true

LD_PRELOAD="/path/to/libtcmalloc.so.4:$LD_PRELOAD" \
    torchrun --nproc_per_node=$NGPU train.py \
    --model resnet101 \
    --dataset tfds/imagenet2012 \
    --dataset-download \
    --data-dir /path/to/tensorflow_datasets \
    --pin-mem \
    --output /path/to/output \
    --experiment "$EXP_NAME" \
    --crop-pct 0.875 \
    --train-interpolation bilinear \
    --resume auto \
    --num-classes 1000 \
    --workers $NCPU \
    --batch-size $LOCAL_BATCH_SIZE \
    --sharpness-m $SHARPNESS_M \
    --grad-accum-steps $((BATCH_SIZE/(NGPU*LOCAL_BATCH_SIZE))) \
    --validation-batch-size 64 \
    --epochs 90 \
    --warmup-epochs 16 \
    --lr 1.6 \
    --min-lr 0.016 \
    --warmup-lr 0 \
    --momentum 0.9 \
    --weight-decay 0.001 \
    --gsam-alpha 0 \
    --min-rho 0.02 \
    --rho 0.04 \
    --obf-C 1000 \
    $@

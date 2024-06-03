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
export BATCH_SIZE=256

export OMP_NUM_THREADS=$((NCPU*2))
export TFDS_TP_SIZE=$((NCPU*2))
export TFDS_SHUFFLE_SIZE=8192
export TFDS_AUTOTUNE_BYTES=$((1024*1024*1024*2))
export NO_GCE_CHECK=true

export USE_FUSED_ATTN=0

echo "========================================="
echo "Experiment: $EXP_NAME"
echo "========================================="
LD_PRELOAD="/path/to/libtcmalloc.so.4:$LD_PRELOAD" \
    python -u validate.py \
    --validate-hessian \
    --dataset tfds/imagenet2012 \
    --split "train[:8192]" \
    --model resnet50 \
    --img-size 224 \
    --data-dir /path/to/tensorflow_datasets \
    --pin-mem \
    --crop-pct 0.875 \
    --train-interpolation bilinear \
    --num-classes 1000 \
    --workers $NCPU \
    --batch-size $BATCH_SIZE \
    --checkpoint /path/to/output/$EXP_NAME/last.pth.tar \
    $@

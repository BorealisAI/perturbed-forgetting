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

export OMP_NUM_THREADS=$((NCPU*2))
export TFDS_TP_SIZE=$((NCPU*2))
export TFDS_AUTOTUNE_BYTES=$((1024*1024*1024*2))
export NO_GCE_CHECK=true

for eval_config in \
    "--dataset tfds/cifar$NUM_CLASSES --split test";
do
    echo "========================================="
    echo "Experiment: $EXP_NAME"
    echo "Evalution: $eval_config"
    echo "========================================="
    LD_PRELOAD="/path/to/libtcmalloc.so.4:$LD_PRELOAD" \
        python validate.py \
        --model vit_small_patch32_224 \
        --model-kwargs class_token=False \
        --data-dir /path/to/tensorflow_datasets \
        --pin-mem \
        --crop-pct 0.875 \
        --num-classes $NUM_CLASSES \
        --workers $NCPU \
        --batch-size 64 \
        --gp avg \
        --log-freq 150 \
        --dataset-download \
        --checkpoint /path/to/output/$EXP_NAME/last.pth.tar \
        $eval_config
done

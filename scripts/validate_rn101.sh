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

export OMP_NUM_THREADS=$((NCPU*2))
export TFDS_TP_SIZE=$((NCPU*2))
export TFDS_AUTOTUNE_BYTES=$((1024*1024*1024*2))
export NO_GCE_CHECK=true

for eval_config in \
    "--dataset tfds/imagenet2012 --split validation" \
    "--dataset tfds/imagenet2012_real --split validation" \
    "--dataset tfds/imagenet_v2 --split test" \
    "--dataset tfds/imagenet_r --split test" \
    "--dataset tfds/imagenet_sketch --split test";
do
    echo "========================================="
    echo "Experiment: $EXP_NAME"
    echo "Evalution: $eval_config"
    echo "========================================="
    LD_PRELOAD="/path/to/libtcmalloc.so.4:$LD_PRELOAD" \
        python validate.py \
        --model resnet101 \
        --data-dir /path/to/tensorflow_datasets \
        --pin-mem \
        --crop-pct 0.875 \
        --num-classes 1000 \
        --workers $NCPU \
        --batch-size 64 \
        --log-freq 150 \
        --dataset-download \
        --checkpoint /path/to/output/$EXP_NAME/last.pth.tar \
        $eval_config
done

#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
###############################################################

set -e
export NGPU=${NGPU:-1}
export NCPU=${NCPU:-4}

####### ViT-S/32 experiments #######

# Training on ImageNet
EXP_NAME=vits32_vanilla ./train_vits32.sh --opt adamw
EXP_NAME=vits32_labelsmooth ./train_vits32.sh --opt adamw --smoothing 0.1
EXP_NAME=vits32_shrink_perturb ./train_vits32.sh --opt adamw --snp-factor 1e-5
EXP_NAME=vits32_sam_max ./train_vits32.sh --opt sam_adamw --rho-policy lr_prop --perturb-loss maximize
EXP_NAME=vits32_sam_obf ./train_vits32.sh --opt sam_adamw --perturb-loss obf
EXP_NAME=vits32_sam_rand ./train_vits32.sh --opt randsam_adamw --perturb-loss maximize
EXP_NAME=vits32_gsam_max ./train_vits32.sh --opt sam_adamw --gsam-alpha 0.4 --rho-policy lr_prop --perturb-loss maximize
EXP_NAME=vits32_gsam_obf ./train_vits32.sh --opt sam_adamw --gsam-alpha 0.4 --backup-normalized --perturb-loss obf
EXP_NAME=vits32_asam_max ./train_vits32.sh --opt sam_adamw --adaptive-sam --asam-after-norm --rho 6.0 --perturb-loss maximize
EXP_NAME=vits32_asam_obf ./train_vits32.sh --opt sam_adamw --adaptive-sam --asam-after-norm --rho 6.0 --perturb-loss obf --obf-gamma 1e-12

# Finetuning on CIFAR-10
NUM_CLASSES=10 EXP_NAME=vits32c10_max2sgd ./finetune_vits32_cifar.sh --opt sgd --pretrained-path /path/to/output/vits32_sam_max/last.pth.tar
NUM_CLASSES=10 EXP_NAME=vits32c10_obf2sgd ./finetune_vits32_cifar.sh --opt sgd --pretrained-path /path/to/output/vits32_sam_obf/last.pth.tar
NUM_CLASSES=10 EXP_NAME=vits32c10_adamw2max ./finetune_vits32_cifar.sh --opt sam_sgd --perturb-loss maximize --pretrained-path /path/to/output/vits32_vanilla/last.pth.tar
NUM_CLASSES=10 EXP_NAME=vits32c10_adamw2obf ./finetune_vits32_cifar.sh --opt sam_sgd --perturb-loss obf --obf-gamma 1e-12 --pretrained-path /path/to/output/vits32_vanilla/last.pth.tar

# Finetuning on CIFAR-100
NUM_CLASSES=100 EXP_NAME=vits32c100_max2sgd ./finetune_vits32_cifar.sh --opt sgd --pretrained-path /path/to/output/vits32_sam_max/last.pth.tar
NUM_CLASSES=100 EXP_NAME=vits32c100_obf2sgd ./finetune_vits32_cifar.sh --opt sgd --pretrained-path /path/to/output/vits32_sam_obf/last.pth.tar
NUM_CLASSES=100 EXP_NAME=vits32c100_adamw2max ./finetune_vits32_cifar.sh --opt sam_sgd --perturb-loss maximize --pretrained-path /path/to/output/vits32_vanilla/last.pth.tar
NUM_CLASSES=100 EXP_NAME=vits32c100_adamw2obf ./finetune_vits32_cifar.sh --opt sam_sgd --perturb-loss obf --obf-gamma 1e-12 --pretrained-path /path/to/output/vits32_vanilla/last.pth.tar

####### ResNet-50 experiments #######

# Training on ImageNet
EXP_NAME=rn50_vanilla ./train_rn50.sh --opt sgdw
EXP_NAME=rn50_labelsmooth ./train_rn50.sh --opt sgdw --smoothing 0.1
EXP_NAME=rn50_shrink_perturb ./train_rn50.sh --opt sgdw --snp-factor 1e-5
EXP_NAME=rn50_sam_max ./train_rn50.sh --opt sam_sgdw --rho-policy lr_prop --perturb-loss maximize
EXP_NAME=rn50_sam_obf ./train_rn50.sh --opt sam_sgdw --perturb-loss obf --obf-gamma 1e-12
EXP_NAME=rn50_sam_rand ./train_rn50.sh --opt randsam_sgdw --perturb-loss maximize
EXP_NAME=rn50_gsam_max ./train_rn50.sh --opt sam_sgdw --gsam-alpha 0.01 --backup-normalized --rho-policy lr_prop --perturb-loss maximize
EXP_NAME=rn50_gsam_obf ./train_rn50.sh --opt sam_sgdw --gsam-alpha 0.01 --perturb-loss obf --obf-gamma 1e-12
EXP_NAME=rn50_asam_max ./train_rn50.sh --opt sam_sgdw --adaptive-sam --asam-before-norm --rho 0.8 --min-rho 0.4 --rho-policy lr_prop --perturb-loss maximize
EXP_NAME=rn50_asam_obf ./train_rn50.sh --opt sam_sgdw --adaptive-sam --asam-before-norm --rho 0.8 --perturb-loss obf --obf-gamma 1e-12

# Finetuning on CIFAR-10
NUM_CLASSES=10 EXP_NAME=rn50c10_max2sgd ./finetune_rn50_cifar.sh --opt sgd --pretrained-path /path/to/output/rn50_sam_max/last.pth.tar
NUM_CLASSES=10 EXP_NAME=rn50c10_obf2sgd ./finetune_rn50_cifar.sh --opt sgd --pretrained-path /path/to/output/rn50_sam_obf/last.pth.tar
NUM_CLASSES=10 EXP_NAME=rn50c10_sgd2max ./finetune_rn50_cifar.sh --bce-loss --opt sam_sgd --perturb-loss maximize --pretrained-path /path/to/output/rn50_vanilla/last.pth.tar
NUM_CLASSES=10 EXP_NAME=rn50c10_sgd2obf ./finetune_rn50_cifar.sh --bce-loss --opt sam_sgd --perturb-loss obf --obf-gamma 1e-12 --pretrained-path /path/to/output/rn50_vanilla/last.pth.tar

# Finetuning on CIFAR-100
NUM_CLASSES=100 EXP_NAME=rn50c100_max2sgd ./finetune_rn50_cifar.sh --opt sgd --pretrained-path /path/to/output/rn50_sam_max/last.pth.tar
NUM_CLASSES=100 EXP_NAME=rn50c100_obf2sgd ./finetune_rn50_cifar.sh --opt sgd --pretrained-path /path/to/output/rn50_sam_obf/last.pth.tar
NUM_CLASSES=100 EXP_NAME=rn50c100_sgd2max ./finetune_rn50_cifar.sh --bce-loss --opt sam_sgd --perturb-loss maximize --pretrained-path /path/to/output/rn50_vanilla/last.pth.tar
NUM_CLASSES=100 EXP_NAME=rn50c100_sgd2obf ./finetune_rn50_cifar.sh --bce-loss --opt sam_sgd --perturb-loss obf --obf-gamma 1e-12 --pretrained-path /path/to/output/rn50_vanilla/last.pth.tar

####### ViT-S/16 experiments #######

# Training on ImageNet
EXP_NAME=vits16_vanilla ./train_vits16.sh --opt adamw
EXP_NAME=vits16_sam_max ./train_vits16.sh --opt sam_adamw --perturb-loss maximize
EXP_NAME=vits16_sam_obf ./train_vits16.sh --opt sam_adamw --perturb-loss obf
EXP_NAME=vits16_gsam_max ./train_vits16.sh --opt sam_adamw --gsam-alpha 1.0 --rho-policy lr_prop --perturb-loss maximize
EXP_NAME=vits16_gsam_obf ./train_vits16.sh --opt sam_adamw --gsam-alpha 0.4 --backup-normalized --perturb-loss obf

####### ResNet-101 experiments #######

# Training on ImageNet
EXP_NAME=rn101_vanilla ./train_rn101.sh --opt sgdw
EXP_NAME=rn101_sam_max ./train_rn101.sh --opt sam_sgdw --rho-policy lr_prop --perturb-loss maximize
EXP_NAME=rn101_sam_obf ./train_rn101.sh --opt sam_sgdw --perturb-loss obf --obf-gamma 1e-12
EXP_NAME=rn101_gsam_max ./train_rn101.sh --opt sam_sgdw --gsam-alpha 0.01 --backup-normalized --rho-policy lr_prop --perturb-loss maximize
EXP_NAME=rn101_gsam_obf ./train_rn101.sh --opt sam_sgdw --gsam-alpha 0.01 --perturb-loss obf --obf-gamma 1e-12

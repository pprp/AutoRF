#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb
export PYTHONUNBUFFERED=1

# 模块设计的消融实验 CIFAR100 | RESNET20 

python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/R1_sam_train_cifar100_resnet20_cutout8_epoch200.log"



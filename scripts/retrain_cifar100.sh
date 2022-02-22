#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb
export PYTHONUNBUFFERED=1

python tools/retrain_cifar100.py -model 'rf_resnet56' -arch "P1"  | tee "./exps/retrain_cifar100/retrain_cifar100_resnet56.log"
# python tools/retrain_cifar100.py -model 'rf_resnet20' -arch "P1"  | tee "./exps/retrain_cifar100/retrain_cifar100_resnet20.log"
#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb
export PYTHONUNBUFFERED=1

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "P1" --rho 1.0 | tee "./exps/sam_train/sam_train_cifar100_resnet56.log"

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "P1" --rho 1.0 | tee "./exps/sam_train/sam_train_cifar100_resnet56_cutout8.log"

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "P1" --rho 1.0 --epochs 600 | tee "./exps/sam_train/sam_train_cifar100_resnet56_cutout8_epoch600.log"

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "P1" --rho 1.0 --epochs 500 | tee "./exps/sam_train/sam_train_cifar100_resnet20_cutout8_epoch500.log"

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "P1" --rho 1.0 --epochs 500 | tee "./exps/sam_train/sam_train_cifar100_resnet32_cutout8_epoch500.log"


# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/sam_train/sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/sam_train/sam_train_cifar100_resnet32_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/sam_train/sam_train_cifar100_resnet56_cutout8_epoch200.log"

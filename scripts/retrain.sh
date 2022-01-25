#!/bin/bash 

# python tools/retrain.py --arch RFSTEP3 --cutout --cutout_length 8 --model_name "retain_rf_resnet_cutout_8"


# python tools/retrain.py --arch Attention  --model_name "autola_base"

python tools/retrain.py --model_base 'resnet20' --model_name 'resnet20_base' --arch 'NORMAL'

# module load  cuda/10.1
# module load cudnn/7.6.5.32_cuda10.1u2

# module load anaconda
# source activate torch17 

# 1.20日 18:30开始运行程序
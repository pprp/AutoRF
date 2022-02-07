#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb
export PYTHONUNBUFFERED=1


# python tools/retrain.py --model_base 'resnet20' --model_name 'resnet20_base' --arch 'NORMAL' & \
# python tools/retrain.py --model_base 'resnet32' --model_name 'resnet32_base' --arch 'NORMAL' & \
# python tools/retrain.py --model_base 'resnet44' --model_name 'resnet44_base' --arch 'NORMAL'

# python tools/retrain.py --model_base 'resnet56' --model_name 'resnet56_base' --arch 'NORMAL' & \
# python tools/retrain.py --model_base 'resnet110' --model_name 'resnet110_base' --arch 'NORMAL'

# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_base_rfstep3' --arch RFSTEP3 
# & \
# python tools/retrain.py --model_base 'rf_resnet32' --model_name 'rf_resnet20_base_rfstep3' --arch RFSTEP3

# python tools/retrain.py --model_base 'rf_resnet44' --model_name 'rf_resnet44_base_rfstep3' --arch RFSTEP3 & \
# python tools/retrain.py --model_base 'rf_resnet56' --model_name 'rf_resnet56_base_rfstep3' --arch RFSTEP3

# python tools/retrain.py --model_base 'rf_resnet110' --model_name 'rf_resnet110_base_rfstep3' --arch RFSTEP3 

# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_base_rfstep3' --arch Attention

# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_labelsmooth_cutout8' --arch Attention --cutout --cutout_length 8

# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_wd2e3_ricap' --arch Attention   --weight_decay 2e-3 

# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_cutout8_wd5e4' --arch Attention --cutout --cutout_length 8 --weight_decay 5e-4 

# python tools/retrain.py --model_base 'la_resnet32' --model_name 'la_resnet20_base_rfstep3' --arch Attention & \
# python tools/retrain.py --model_base 'la_resnet44' --model_name 'la_resnet44_base_rfstep3' --arch Attention

# python tools/retrain.py --model_base 'la_resnet56' --model_name 'la_resnet56_base_rfstep3' --arch Attention & \
# python tools/retrain.py --model_base 'la_resnet110' --model_name 'la_resnet110_base_rfstep3' --arch Attention


# p1 = C//4 + SE=True + SPP1(Fullpool)
# p2 = C//4 + SE=False + SPP2(Dilconv)
# P3 = C//4 + SE=False + 3x3(not 1x1) + SPP2(Dilconv)
# p4 = C//4 + SE=True + SPP2(Dilconv)
# p5 = C//4 + SE=True + SPP3(Hybrid conv+pool)

# P4
# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P4' --arch P4

# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P3' --arch P3

# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P2' --arch P2

# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P1' --arch P1 

# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P5' --arch P5

python tools/retrain.py --model_base 'rfsa_resnet20' --model_name 'rfsa_resnet20_P6' --arch P6

# module load  cuda/10.1
# module load cudnn/7.6.5.32_cuda10.1u2

# module load anaconda
# source activate torch17 

# 1.20日 18:30开始运行程序
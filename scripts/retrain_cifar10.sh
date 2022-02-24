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

# AUTO LEARNING ATTENTION   调参
# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_cutout8_lr0.2' --arch Attention --cutout --cutout_length 8 --learning_rate 0.2
# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_cutout8_lr0.05' --arch Attention --cutout --cutout_length 8 --learning_rate 0.05
# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_cutout8_lr0.025' --arch Attention --cutout --cutout_length 8 --learning_rate 0.025
# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_cutout8_bs=128_dropout=0.5' --arch Attention --cutout --cutout_length 8 --batch_size 128 
# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_cutout8_bs=64' --arch Attention --cutout --cutout_length 8 --batch_size 64 
# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_wd2e3_ricap' --arch Attention   --weight_decay 2e-3 
# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_cutout8_wd5e4' --arch Attention --cutout --cutout_length 8 --weight_decay 5e-4 
# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_cutout8_bs128_wd2e3' --arch Attention --cutout --cutout_length 8 --batch_size 128 --weight_decay 2e-3 
# python tools/retrain.py --model_base 'la_resnet32' --model_name 'la_resnet20_base_rfstep3' --arch Attention & \
# python tools/retrain.py --model_base 'la_resnet44' --model_name 'la_resnet44_base_rfstep3' --arch Attention
# python tools/retrain.py --model_base 'la_resnet56' --model_name 'la_resnet56_base_rfstep3' --arch Attention & \
# python tools/retrain.py --model_base 'la_resnet110' --model_name 'la_resnet110_base_rfstep3' --arch Attention


# p1 = C//4 + SE=True + SPP1(Fullpool)
# p2 = C//4 + SE=False + SPP2(Dilconv)
# P3 = C//4 + SE=False + 3x3(not 1x1) + SPP2(fgt)
# p4 = C//4 + SE=True + SPP2(Dilconv)
# p5 = C//4 + SE=True + SPP3(Hybrid conv+pool)

# P4
# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P4' --arch P4
# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P3' --arch P3
# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P2' --arch P2
# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P1' --arch P1 
# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P5' --arch P5
# python tools/retrain.py --model_base 'rfsa_resnet20' --model_name 'rfsa_resnet20_P6' --arch P6

# module load  cuda/10.1
# module load cudnn/7.6.5.32_cuda10.1u2

# module load anaconda
# source activate torch17 


# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_cutout8_bs=128_dropout=0.3' --arch P1 --cutout --cutout_length 8 --batch_size 128 

# P6: 修改正确以后的模型
# python tools/retrain.py --model_base 'rfsa_resnet20' --model_name 'p6_rfsa_resnet20_cutout8' --arch P6 --cutout --cutout_length 8 
# python tools/retrain.py --model_base 'rfconvnext_resnet20' --model_name 'p6_rfconvnext_resnet20_cutout8_correct' --arch P6 --cutout --cutout_length 8 
# python tools/retrain.py --model_base 'rfsa_resnet20' --model_name 'p6_rfsa_resnet20_cutout8_bs128' --arch P6 --cutout --cutout_length 8 --batch_size 128 
# python tools/retrain.py --model_base 'rfconvnext_resnet20' --model_name 'p6_rfconvnext_resnet20_cutout8_bs128' --arch P6 --cutout --cutout_length 8 --batch_size 128 

# 复现最好的结果
# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_P1_cutout8_ls' --arch P1 --cutout --cutout_length 8

# 86%
# python tools/retrain.py --model_base 'rfconvnext_resnet20' --model_name 'p6_rfconvnext_resnet20_cutout8_correct' --arch P6 --cutout --cutout_length 8 

# 82% 
# python tools/retrain.py --model_base 'rfconvnext_resnet20' --model_name 'p6_rfconvnext_resnet20_cutout8_correct_bs128' --arch P6 --cutout --cutout_length 8 --batch_size 128 


# 训练resnet20以外的方案 

# 运行中
python tools/retrain.py --model_base 'rf_resnet56' --dataset "cifar100" --model_name 'tttt_rf_resnet20_P1_cutout8_ls_wo_drop' --arch P1 --cutout --cutout_length 8  

# python tools/retrain.py --model_base 'rf_resnet32' --model_name 'rf_resnet32_P1_cutout8_ls' --arch P1 --cutout --cutout_length 8 
# python tools/retrain.py --model_base 'rf_resnet56' --model_name 'rf_resnet56_P1_cutout8_ls' --arch P1 --cutout --cutout_length 8 

# 运行中 
# python tools/retrain.py --model_base 'rf_resnet32' --model_name 'rf_resnet32_P1_cutout8' --arch P1 --cutout --cutout_length 8 & \
# python tools/retrain.py --model_base 'rf_resnet56' --model_name 'rf_resnet56_P1_cutout8' --arch P1 --cutout --cutout_length 8 

# 运行中
# python tools/retrain.py --model_base 'rf_resnet32' --model_name 'rf_resnet32_P1_cutout8_bs128' --arch P1 --cutout --cutout_length 8 --batch_size 128 & \
# python tools/retrain.py --model_base 'rf_resnet56' --model_name  'rf_resnet56_P1_cutout8_bs128' --arch P1 --cutout --cutout_length 8 --batch_size 128


# python tools/retrain.py --model_base 'rf_resnet32' --model_name 'rf_resnet32_P1_cutout8_bs128_asam' --arch P1 --cutout --cutout_length 8 --minimizer "ASAM" --batch_size 128


#########################################################################

# CIFAR100 数据集相关实验 
# 运行中
# resnet20 
# python tools/retrain.py --model_base 'rf_resnet20' \
#                         --dataset 'cifar100' \
#                         --arch P1 \
#                         --model_name 'rf_resnet20_cifar100_none' & \
# python tools/retrain.py --model_base 'rf_resnet20' \
#                         --dataset 'cifar100' \
#                         --cutout --cutout_length 8 \
#                         --arch P1 \
#                         --model_name 'rf_resnet20_cifar100_cutout8'
                        
# python tools/retrain.py --model_base 'rf_resnet20' \
#                         --dataset 'cifar100' \
#                         --label_smooth \
#                         --arch P1 \
#                         --model_name 'rf_resnet20_cifar100_ls' & \
# python tools/retrain.py --model_base 'rf_resnet20' \
#                         --dataset 'cifar100' \
#                         --dropout 0.3 \
#                         --arch P1 \
#                         --model_name 'rf_resnet20_cifar100_drop0.3'

# python tools/retrain.py --model_base 'rf_resnet20' \
#                         --dataset 'cifar100' \
#                         --cutout --cutout_length 8 \
#                         --arch P1 \
#                         --label_smooth \
#                         --batch_size 128 \
#                         --model_name 'rf_resnet20_cifar100_cutout8_bs128_ls'

# python tools/retrain.py --model_base 'rf_resnet20' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --epochs 200 \
#                         --scheduler 'steplr' \
#                         --model_name 'rf_resnet20_cifar100_cutout8_bs128_ls_steplr' 

# python tools/retrain.py --model_base 'rf_resnet20' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --epochs 200 \
#                         --scheduler 'warmup' \
#                         --model_name 'rf_resnet20_cifar100_cutout8_bs128_ls_warmup' 

# python tools/retrain.py --model_base 'rf_resnet20' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --epochs 200 \
#                         --scheduler 'warmup' \
#                         --no_bias_decay \
#                         --model_name 'rf_resnet20_cifar100_cutout8_bs128_ls_warmup_no_bias_decay' 

# resnet32 ############################################################################
# 运行中
# python tools/retrain.py --model_base 'rf_resnet32' \
#                         --dataset 'cifar100' \
#                         --arch P1 \
#                         --model_name 'rf_resnet32_cifar100_none' & \
# python tools/retrain.py --model_base 'rf_resnet32' \
#                         --dataset 'cifar100' \
#                         --cutout --cutout_length 8 \
#                         --arch P1 \
#                         --model_name 'rf_resnet32_cifar100_cutout8'
                        
# python tools/retrain.py --model_base 'rf_resnet32' \
#                         --dataset 'cifar100' \
#                         --label_smooth \
#                         --arch P1 \
#                         --model_name 'rf_resnet32_cifar100_ls' & \
# python tools/retrain.py --model_base 'rf_resnet32' \
#                         --dataset 'cifar100' \
#                         --dropout 0.3 \
#                         --arch P1 \
#                         --model_name 'rf_resnet32_cifar100_drop0.3'


# python tools/retrain.py --model_base 'rf_resnet32' \
#                         --dataset 'cifar100' \
#                         --cutout --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --model_name 'rf_resnet32_cifar100_cutout8_bs128'

# python tools/retrain.py --model_base 'rf_resnet32' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --epochs 200 \
#                         --scheduler 'steplr' \
#                         --model_name 'rf_resnet32_cifar100_cutout8_bs128_ls_steplr' 

# python tools/retrain.py --model_base 'rf_resnet32' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --epochs 200 \
#                         --scheduler 'warmup' \
#                         --model_name 'rf_resnet32_cifar100_cutout8_bs128_ls_warmup' 

# python tools/retrain.py --model_base 'rf_resnet32' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --epochs 200 \
#                         --scheduler 'warmup' \
#                         --no_bias_decay \
#                         --model_name 'rf_resnet32_cifar100_cutout8_bs128_ls_warmup_no_bias_decay' 

# resnet 56 ############################################################################

# python tools/retrain.py --model_base 'rf_resnet56' \
#                         --dataset 'cifar100' \
#                         --arch P1 \
#                         --model_name 'rf_resnet56_cifar100_none' & \
# python tools/retrain.py --model_base 'rf_resnet56' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --model_name 'rf_resnet56_cifar100_cutout8'
                        
# python tools/retrain.py --model_base 'rf_resnet56' \
#                         --dataset 'cifar100' \
#                         --label_smooth \
#                         --arch P1 \
#                         --model_name 'rf_resnet56_cifar100_ls' & \
# python tools/retrain.py --model_base 'rf_resnet56' \
#                         --dataset 'cifar100' \
#                         --dropout 0.3 \
#                         --arch P1 \
#                         --model_name 'rf_resnet56_cifar100_drop0.3'

# python tools/retrain.py --model_base 'rf_resnet56' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --model_name 'rf_resnet56_cifar100_cutout8_bs128_ls'

# python tools/retrain.py --model_base 'rf_resnet56' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --epochs 200 \
#                         --scheduler 'steplr' \
#                         --model_name 'rf_resnet56_cifar100_cutout8_bs128_ls_steplr' 

# python tools/retrain.py --model_base 'rf_resnet56' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --epochs 200 \
#                         --scheduler 'warmup' \
#                         --model_name 'rf_resnet56_cifar100_cutout8_bs128_ls_warmup' 

# python tools/retrain.py --model_base 'rf_resnet56' \
#                         --dataset 'cifar100' \
#                         --cutout \
#                         --cutout_length 8 \
#                         --arch P1 \
#                         --batch_size 128 \
#                         --label_smooth \
#                         --epochs 600 \
#                         --scheduler 'cosine' \
#                         --no_bias_decay \
#                         --learning_rate 0.04 \
#                         --model_name 'cifar100_rf_resnet56_cutout8_bs128_ls_cosine_no_bias_decay_lr0.04' 





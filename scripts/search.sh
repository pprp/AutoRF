#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb

export PYTHONUNBUFFERED=1


# python tools/search.py --model_name rf_p1 
# python tools/search.py --model_name rf_p2 
# python tools/search.py --model_name rf_p3 
# python tools/search.py --model_name rf_p4
# python tools/search.py --model_name rf_p5
# python tools/search.py --model_name rf_p6



# p1 = C//4 + SE=True + SPP1(Fullpool)
# p2 = C//4 + SE=False + SPP2(Dilconv)
# P3 = C//4 + SE=False + 3x3(not 1x1) + SPP2(Dilconv)
# p4 = C//4 + SE=True + SPP2(Dilconv)
# p5 = C//4 + SE=True + SPP3(Hybrid conv+pool)
# p6 = C//4 + SE=True + SPP1(FULLPool) + Self-Attention


# cifar100 
# python tools/search.py --dataset 'cifar100' --model_name "cifar100_rf" --cutout --cutout_length 8 

# imagenet-mini mobilenetv2 rf
# python tools/search.py --dataset 'imagenet' --data '/data/public/imagenet-mini' --model_name "mobilenetv2_imagenet_mini" --model_name 'mobilenet_rf_v2'

date 
# cp -r /data/public/imagenet-mini /dev/shm
unzip /HOME/scz0088/run/datasets/imagenet-mini.zip -d /dev/shm/imagenet-mini
date 


# imagenet-mini mobilentv2 rfsa 
python tools/search.py --dataset 'imagenet' --data '/dev/shm/imagenet-mini' --model_name "mobilenetv2_imagenet_mini" --model_name 'mobilenet_rfsa_v2' --batch_size 64 

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

# date 
# # cp -r /data/public/imagenet-mini /dev/shm
# unzip -o /HOME/scz0088/run/datasets/imagenet-mini.zip -d /dev/shm/imagenet-mini > /dev/null
# date 


# # imagenet-mini mobilentv2 rfsa 
# python tools/search.py --dataset 'imagenet' --data '/dev/shm/imagenet-mini' --model_name 'mobilenet_rfsa_v2' --batch_size 64 

########## ablation study ###########
# 注意：修改spaces中的PRITMITIVE, 然后运行试验
# 注意：修改model_name名称 设置为resnet20模式

# cifar100 small primitive
# python tools/search.py --dataset 'cifar100' --model_name "rf_resnet20" --cutout --cutout_length 8 --comments "small_primitive" --primitives "small"

# cifar100 middle primitive
# python tools/search.py --dataset 'cifar100' --model_name "rf_resnet20" --cutout --cutout_length 8 --comments "middle_primitive" --primitives "middle"

# cifar100 large primitive
# python tools/search.py --dataset 'cifar100' --model_name "rf_resnet20" --cutout --cutout_length 8 --comments "large_primitive" --primitives "large"


############# search in imagenet-mini resnet34, resnet18 rf ###########
# python tools/search.py --dataset 'imagenet' --data '/dev/shm/imagenet-mini/train' --model_name 'resnet34_rf' --batch_size 32  --primitives "fullpool" --learning_rate 0.0125 --comments "imagenet-mini" 

# python tools/search.py --dataset 'imagenet' --data '/dev/shm/imagenet-mini/train' --model_name 'resnet34_rf' --batch_size 32  --primitives "fullpool" --learning_rate 0.0125 --comments "imagenet-mini" 

# 测试不同的数据集，看看是否是由于imagenet-mini过于简单导致的
# python tools/search.py --dataset 'cifar10' --data '/data/public/cifar' --model_name 'resnet34_rf' --batch_size 256  --primitives "fullpool" --learning_rate 0.05 --comments "cifar10_test_resnet34_rf" --arch_learning_rate 1e-4 

# python tools/search.py --dataset 'cifar10' --data '/data/public/cifar' --model_name 'resnet18_rf' --batch_size 256  --primitives "fullpool" --learning_rate 0.05 --comments "cifar10_test_resnet18_rf" --arch_learning_rate 1e-4 

# 依然采用imagenet mini数据集，通过使用noise的方式，防止collapse of darts, 增大noise, 降低arch learning rate 
python tools/search.py --dataset 'imagenet' --data '/data/public/imagenet-mini/train' --model_name 'resnet18_rf' --batch_size 32  --primitives "fullpool" --learning_rate 0.025 --comments "imagenet-mini-resnet18_rf" --arch_learning_rate 5e-5 

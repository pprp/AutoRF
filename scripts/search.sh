#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb

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
python tools/search.py --dataset 'cifar100' --model_name "cifar100_rf" --cutout --cutout_length 8 
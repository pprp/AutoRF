#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb

python tools/search.py --model_name rf_p1 

# p1 = C//4 + SE=True + SPP1(Fullpool)
# p2 = C//4 + SE=False + SPP1(Fullpool)
# P3 = C//4 + SE=False + 3x3 + SPP1(FullPool)


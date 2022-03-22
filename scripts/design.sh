#!/bin/bash 

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb
export PYTHONUNBUFFERED=1


# 注意：修改搜索空间的时候记得确认spaces.py最后一行配置。 
# 注意：针对cifar10和cifar100的结果需要修改studentnet.py中的内容。

# ablation study for random search 

### CIFAR10 ablation study for random search 
# cancel.....
# python tools/sam_train.py --dataset "CIFAR10" --model 'rf_resnet20' --arch "R1" --rho 1.0 --epochs 200 --batch_size 128 | tee "./exps/ablation/R1_sam_train_cifar10_resnet20_cutout8_epoch500_bs128.log"
# python tools/sam_train.py --dataset "CIFAR10" --model 'rf_resnet32' --arch "R1" --rho 1.0 --epochs 200 --batch_size 128 | tee "./exps/ablation/R1_sam_train_cifar10_resnet32_cutout8_epoch500_bs128.log"
# python tools/sam_train.py --dataset "CIFAR10" --model 'rf_resnet56' --arch "R1" --rho 1.0 --epochs 200 --batch_size 128 | tee "./exps/ablation/R1_sam_train_cifar10_resnet56_cutout8_epoch500_bs128.log"


### CIFAR100 ablation study for random search 

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/R1_sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/R1_sam_train_cifar100_resnet32_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/R1_sam_train_cifar100_resnet56_cutout8_epoch200.log"

### CIFAR10 ablation study for search space 



### CIFAR100 ablation study for search space 

# P4 = FULLCONV search space 
# 记得修改spaces.py 
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "P4" --rho 1.0 --epochs 200 | tee "./exps/ablation/P4_R1_sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "P4" --rho 1.0 --epochs 200 | tee "./exps/ablation/P4_sam_train_cifar100_resnet32_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "P4" --rho 1.0 --epochs 200 | tee "./exps/ablation/P4_sam_train_cifar100_resnet56_cutout8_epoch200.log"

# P5 = HYBRID search space 
# 记得修改spaces.py 
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "P5" --rho 1.0 --epochs 200 | tee "./exps/ablation/P5_sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "P5" --rho 1.0 --epochs 200 | tee "./exps/ablation/P5_sam_train_cifar100_resnet32_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "P5" --rho 1.0 --epochs 200 | tee "./exps/ablation/P5_sam_train_cifar100_resnet56_cutout8_epoch200.log"


### ablation study for insert part 

# 已结束：结论：参数量小的模型，插入模块后插入的越多，性能越高；参数量大的模型，插入的越少，性能越高。

# C16 & C64 
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20_ccc' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/ablation/insert_sam_train_cifar100_resnet20_cutout8_epoch200_ccc.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20_cct' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/ablation/insert_sam_train_cifar100_resnet20_cutout8_epoch200_cct.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20_ctt' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/ablation/insert_sam_train_cifar100_resnet20_cutout8_epoch200_ctt.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20_ttt' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/ablation/insert_sam_train_cifar100_resnet20_cutout8_epoch200_ttt.log"


# C32 & C128
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20_ccc' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/ablation/insert_c128_sam_train_cifar100_resnet20_cutout8_epoch200_ccc.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20_cct' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/ablation/insert_c128_sam_train_cifar100_resnet20_cutout8_epoch200_cct.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20_ctt' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/ablation/insert_c128_sam_train_cifar100_resnet20_cutout8_epoch200_ctt.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20_ttt' --arch "P1" --rho 1.0 --epochs 200 | tee "./exps/ablation/insert_c128_sam_train_cifar100_resnet20_cutout8_epoch200_ttt.log"



## 实验进展描述：
## 发现混合搜索空间效果更优，因此随机搜索方法打算在混合空间进行搜索和尝试。
# 注意：修改spaces.py中搜索空间。目前状态为：Hybrid

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN1_R1_sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN1_R1_sam_train_cifar100_resnet32_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN1_R1_sam_train_cifar100_resnet56_cutout8_epoch200.log"

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN2_R1_sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN2_R1_sam_train_cifar100_resnet32_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN2_R1_sam_train_cifar100_resnet56_cutout8_epoch200.log"

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN3_R1_sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN3_R1_sam_train_cifar100_resnet32_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN3_R1_sam_train_cifar100_resnet56_cutout8_epoch200.log"

# 目前状态为：FULLPOOL 

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN4_R1_sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN4_R1_sam_train_cifar100_resnet32_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN4_R1_sam_train_cifar100_resnet56_cutout8_epoch200.log"

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN5_R1_sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN5_R1_sam_train_cifar100_resnet32_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN5_R1_sam_train_cifar100_resnet56_cutout8_epoch200.log"

# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet20' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN6_R1_sam_train_cifar100_resnet20_cutout8_epoch200.log"
# python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet32' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN6_R1_sam_train_cifar100_resnet32_cutout8_epoch200.log"
python tools/sam_train.py --dataset "CIFAR100" --model 'rf_resnet56' --arch "R1" --rho 1.0 --epochs 200 | tee "./exps/ablation/HYBRID_RUN6_R1_sam_train_cifar100_resnet56_cutout8_epoch200.log"


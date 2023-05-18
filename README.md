# AutoRF: Auto Learning Receptive Fields with Spatial Pooling

Accepted by International Conference on MultiMedia Modeling 2023

## Introduction

This repository contains the source code of the paper: AutoRF: Auto Learning Receptive Fields with Spatial Pooling.

## Requirements

- Python 3.7+
- Cuda 11
- tensorboardX
- torch
- torchvision
- graphviz
- numpy
- thop
- timm

## Usage

### 1. Clone the repository

```shell
git clone https://github.com/pprp/AutoRF.git
```

### 2. Search

```shell
python search.py --data /path/to/data --dataset cifar10 --primitives fullpool --model_name rf_p5
```

### 3. Retrain

```shell
python tools/retrain.py --model_base 'rf_resnet20' \
                        --dataset 'cifar100' \
                        --cutout \
                        --cutout_length 8 \
                        --arch P1 \
                        --batch_size 128 \
                        --label_smooth \
                        --epochs 200 \
                        --scheduler 'steplr' \
                        --model_name 'rf_resnet20_cifar100_cutout8_bs128_ls_steplr'
```


## Citation

```
@inproceedings{Dong2023AutoRFAL,
  title={AutoRF: Auto Learning Receptive Fields with Spatial Pooling},
  author={Peijie Dong and Xin Niu and Zimian Wei and Hengyue Pan and Dongsheng Li and Zhen Huang},
  booktitle={Conference on Multimedia Modeling},
  year={2023}
}
```

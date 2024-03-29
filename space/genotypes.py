import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collections import namedtuple

from space.spaces import spatial_spaces

Genotype = namedtuple('Genotype', 'normal normal_concat')

NORMAL = None

Attention = Genotype(
    normal=[
        ('skip_connect', 0),
        ('sep_conv_3x3_spatial', 0),
        ('sep_conv_3x3', 1),
        ('CBAM', 0),
        ('skip_connect', 2),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 1),
        ('sep_conv_3x3', 3),
        ('dil_conv_3x3', 2),
    ],
    normal_concat=range(1, 5),
)

Attention_Searched = Genotype(
    normal=[
        ('skip_connect', 0),
        ('CBAM', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('dil_conv_3x3_spatial', 2),
        ('CBAM', 0),
        ('skip_connect', 1),
        ('CBAM', 0),
        ('dil_conv_3x3', 2),
        ('SE_A_M', 3),
    ],
    normal_concat=range(1, 5),
)

# for cifar10
P1 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0),
                      ('max_pool_5x5', 1), ('max_pool_3x3', 0), ('noise', 2),
                      ('noise', 1)],
              normal_concat=range(0, 4))

# P2 =  Genotype(normal=[('conv_5x1_1x5', 0), ('dil_conv_3x3', 1), ('conv_3x1_1x3', 0), ('dil_conv_3x3_spatial', 1), ('dil_conv_5x5', 2), ('conv_5x1_1x5', 0)], normal_concat=range(0, 4))

# P3 = Genotype(normal=[('dil_conv_3x3_spatial', 0), ('dil_conv_3x3', 1), ('conv_5x1_1x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('conv_5x1_1x5', 0)], normal_concat=range(0, 4))

# for cifar10 FULLCONV search space
# P4 = Genotype(normal=[('conv_5x1_1x5', 0), ('sep_conv_3x3_spatial', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3_spatial', 1), ('sep_conv_3x3_spatial', 2), ('conv_3x1_1x3', 0)], normal_concat=range(0, 4))

# for cifar10 Hybrid search space
# P5 = Genotype(normal=[('avg_pool_3x3', 0), ('conv_3x1_1x3', 0), ('sep_conv_3x3_spatial', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3_spatial', 1), ('noise', 2)], normal_concat=range(0, 4))

# for cifar100
Q1 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_5x5', 0),
                      ('max_pool_7x7', 1), ('noise', 1), ('noise', 2),
                      ('noise', 0)],
              normal_concat=range(0, 4))

# for mobilentv2
M1 = Genotype(normal=[('strippool', 0),
                      ('avg_pool_3x3', 0), ('avg_pool_5x5', 1),
                      ('avg_pool_7x7', 0), ('strippool', 2), ('noise', 1)],
              normal_concat=range(0, 4))

# for resnet18 imagenet-mini
M2 = Genotype(normal=[('avg_pool_7x7', 0), ('noise', 0), ('noise', 1),
                      ('noise', 1), ('noise', 2), ('noise', 0)],
              normal_concat=range(0, 4))

# for resnet18 cifar100
M3 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_7x7', 1),
                      ('avg_pool_7x7', 0), ('max_pool_5x5', 1),
                      ('max_pool_3x3', 0), ('max_pool_5x5', 2)],
              normal_concat=range(0, 4))

# random search for ablation study
PRIMITIVES = spatial_spaces['small']  # TODO more aligent way
R1 = Genotype(normal=[(random.choice(PRIMITIVES), 0),
                      (random.choice(PRIMITIVES), random.choice([0, 1])),
                      (random.choice(PRIMITIVES), random.choice([0, 1])),
                      (random.choice(PRIMITIVES), random.choice([0, 1, 2])),
                      (random.choice(PRIMITIVES), random.choice([0, 1, 2])),
                      (random.choice(PRIMITIVES), random.choice([0, 1, 2]))],
              normal_concat=range(0, 4))

# for small middle large search space

SMALL = Genotype(normal=[('avg_pool_3x3', 0), ('noise', 1), ('noise', 0),
                         ('noise', 2), ('noise', 1), ('noise', 0)],
                 normal_concat=range(0, 4))

MIDDLE = Genotype(normal=[('avg_pool_3x3', 0),
                          ('avg_pool_3x3', 0), ('noise', 1), ('noise', 2),
                          ('noise', 1), ('noise', 0)],
                  normal_concat=range(0, 4))

LARGE = Genotype(normal=[('avg_pool_3x3', 0),
                         ('max_pool_3x3', 0), ('noise', 1), ('noise', 2),
                         ('noise', 1), ('noise', 0)],
                 normal_concat=range(0, 4))

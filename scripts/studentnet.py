import torch
import torch.nn as nn
import torch.nn.functional as F
# from space.operations import *
from torch.autograd import Variable
# from space.spaces import PRIMITIVES
# from space.genotypes import Genotype
import retrain.basemodel as bm
# import basemodel as bm


class Network(nn.Module):
    def __init__(self, model_base, num_classes, genotype, dropout=0.):
        super(Network, self).__init__()
        self._num_classes = num_classes
        self.genotype = genotype
        self.model = bm.__dict__[model_base](num_classes=self._num_classes,
                                             genotype=self.genotype, dropout=dropout)

    def forward(self, x):
        return self.model(x)


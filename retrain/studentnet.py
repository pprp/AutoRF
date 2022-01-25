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
    def __init__(self, model_base, num_classes, genotype):
        super(Network, self).__init__()
        self._num_classes = num_classes
        self.genotype = genotype
        self.model = bm.__dict__[model_base](num_classes=self._num_classes,
                                             genotype=self.genotype)

    def forward(self, x):
        return self.model(x)

ms = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'rf_resnet20', 'rf_resnet32', 'rf_resnet44',
      'rf_resnet56', 'rf_resnet110', 'la_resnet20', 'la_resnet32', 'la_resnet44', 'la_resnet56', 'la_resnet110']

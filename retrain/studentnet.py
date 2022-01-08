import torch
import torch.nn as nn
import torch.nn.functional as F
from space.operations import *
from torch.autograd import Variable
from space.genotypes import PRIMITIVES
from space.genotypes import Genotype
from retrain.basemodel import *


class Network(nn.Module):
    def __init__(self, num_classes, genotype):
        super(Network, self).__init__()
        self._num_classes = num_classes
        self.genotype = genotype
        model = attention_resnet20(
            num_classes=self._num_classes, genotype=self.genotype
        )
        self.model = model

    def forward(self, x):
        return self.model(x)

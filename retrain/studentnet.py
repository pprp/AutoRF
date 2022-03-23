import torch
import torch.nn as nn
import torch.nn.functional as F
# from space.operations import *
from torch.autograd import Variable
import retrain.resnet_c100 as c100
import retrain.resnet_c10 as c10 
import retrain.ablation as ab 

model_list = {**c100.__dict__, **c10.__dict__, **ab.__dict__}

class Network(nn.Module):
    def __init__(self, model_base, num_classes, genotype, dropout=0.):
        super(Network, self).__init__()
        self._num_classes = num_classes
        self.genotype = genotype
        self.model = model_list[model_base](num_classes=self._num_classes,
                                             genotype=self.genotype, dropout=dropout)

    def forward(self, x):
        return self.model(x)


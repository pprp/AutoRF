import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from search.components import *
from space.genotypes import Genotype
from space.operations import *
from space.spaces import spatial_spaces

# from .resnet_cifar import *
from .mobilenetv2 import *
from .resnet_imagenet import *


class Network(nn.Module):
    def __init__(self, num_classes, model_name, primitives='fullpool'):
        '''
        primitive: autola fullpool fullconv hybrid small middle large
        '''

        super(Network, self).__init__()

        self.PRIMITIVES = spatial_spaces[primitives]
        self._num_classes = num_classes
        self._criterion = nn.CrossEntropyLoss().cuda()
        self.model = eval(model_name)(num_classes=num_classes,
                                      PRIMITIVES=self.PRIMITIVES)
        self._steps = 3
        self._multiplier = 4
        self._initialize_alphas()

    def forward(self, x):
        weights = F.softmax(self.alphas_normal, dim=-1)
        return self.model(x, weights)

    def new(self):
        model_new = Network(self._num_classes).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _loss(self, input, target):

        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(1 + i))
        num_ops = len(self.PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(),
                                      requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 1),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                       if k != self.PRIMITIVES.index('none')),
                )[:i + 1]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != self.PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((self.PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())

        concat = range(1 + self._steps - self._multiplier, self._steps + 1)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
        )
        return genotype

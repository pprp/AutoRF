import os
import sys
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrain.basemodel import conv3x3

from space.operations import SE
from space.spaces import OPS

Genotype = namedtuple("Genotype", "normal normal_concat")

class PluginRF(nn.Module):
    def __init__(self, inplanes, steps=3, se=True, genotype=None, spatial=False, channel=False, reduction_ratio=4):
        super(PluginRF, self).__init__()
        assert genotype is not None
        self._se = se
        self.inplanes = inplanes

        # config
        self.spatial = spatial 
        self.channel = channel
        self.reduction_ratio = 16
        self._steps = steps

        # spatial attention 
        if self.spatial:
            self.genotype = genotype
            self._ops = nn.ModuleList()
            op_names, indices = zip(*self.genotype.normal)
            concat = genotype.normal_concat
            self.bottle = nn.Conv2d(inplanes, inplanes // self.reduction_ratio, kernel_size=1,stride=1, padding=0, bias=False)
            self.conv1x1 = nn.Conv2d(inplanes // self.reduction_ratio * self._steps, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
            self._compile(op_names, indices, concat)

        # channel attention 
        if self.channel:
            if self._se: 
                self.se = SE(self.inplanes, reduction=self.reduction_ratio)


    def _compile(self, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, _ in zip(op_names, indices):
            op = OPS[name](self.inplanes // self.reduction_ratio, 1, True)
            self._ops += [op]
        self.indices = indices

    def forward(self, x):
        # spatial processing parts 
        if self.spatial:
            t = self.bottle(x)
            states = [t]
            total_step = (1+self._steps) * self._steps // 2
            for i in range(total_step):
                h = states[self.indices[i]]
                ops = self._ops[i]
                s = ops(h)
                states.append(s)
            node_out = torch.cat(states[-self._steps:], dim=1)
            node_out = self.conv1x1(node_out) 
            node_out = node_out + x
        else:
            node_out = x 

        # channel processing parts
        if self.channel:
            if self._se:
                node_out = self.se(node_out)
            else:
                node_out = node_out 
        return node_out

class PluginBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step, genotype=None, **kwargs):
        super(PluginBasicBlock, self).__init__()
        self._steps = step
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.genotype = genotype

        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = PluginRF(planes, genotype=self.genotype, **kwargs)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out = out + residual
        out = self.relu(out)
        return out

class PluginResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=100, genotype=None, dropout=0., **kwargs):
        super(PluginResNet, self).__init__()
        self.inplane = 64 # 16 for cifar10 64 for cifar100 
        self.genotype = genotype
        self.dropout = dropout
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.channel_in = 64

        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self._step = 3
        self.layer1 = self._make_layer(
            block,
            self.channel_in,
            blocks=n_size,
            stride=1,
            step=self._step,
            genotype=self.genotype,
            kwargs=kwargs,
        )
        self.layer2 = self._make_layer(
            block,
            self.channel_in * 2,
            blocks=n_size,
            stride=2,
            step=self._step,
            genotype=self.genotype,
            kwargs=kwargs,
        )

        self.layer3 = self._make_layer(
            block,
            self.channel_in * 4,
            blocks=n_size,
            stride=2,
            step=self._step,
            genotype=self.genotype,
            kwargs=kwargs,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.channel_in * 4, num_classes)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, step, genotype, kwargs):
        strides = [stride] + [1] * (blocks - 1)
        self.layers = nn.ModuleList()
        for stride in strides:
            Block = block(self.inplane, planes, stride, step, genotype, **kwargs)
            self.layers += [Block]
            self.inplane = planes
        return self.layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for _, layer in enumerate(self.layer1):
            x = layer(x)
        for _, layer in enumerate(self.layer2):
            x = layer(x)
        for _, layer in enumerate(self.layer3):
            x = layer(x)

        x = self.avgpool(x)

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def plug_resnet20(**kwargs):
    model = PluginResNet(PluginBasicBlock, 3, **kwargs)
    return model

def plug_spatial_resnet20(channel=False, spatial=True, **kwargs):
    return plug_resnet20(channel=channel, spatial=spatial, **kwargs)

def plug_channel_resnet20(channel=True, spatial=False, **kwargs):
    return plug_resnet20(channel=channel, spatial=spatial, **kwargs)

def plug_spatial_channel_resnet20(channel=True, spatial=True, **kwargs):
    return plug_resnet20(channel=channel, spatial=spatial, **kwargs)

if __name__ == "__main__":
    # m = plug_spatial_resnet20(genotype=Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_5x5', 0), ('max_pool_7x7', 1), ('noise', 1), ('noise', 2), ('noise', 0)], normal_concat=range(0, 4)))
    m = plug_spatial_channel_resnet20(genotype=Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_5x5', 0), ('max_pool_7x7', 1), ('noise', 1), ('noise', 2), ('noise', 0)], normal_concat=range(0, 4)))
    print(m)
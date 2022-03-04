from search.utils import DropPath
from utils.utils import drop_path
from torchvision.models import ResNet
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.autograd import Variable
from space.spaces import OPS
from space.operations import *
import torch.nn as nn
import torch
import os
import pdb
import sys
from collections import namedtuple
import warnings

from retrain.basemodel import *


class InsertResNet(nn.Module):
    '''
    insert attention module 
    C C C: 全部是普通卷积 
    C C T: 最后一个加上Attention 
    C T T: 两个attention 
    T T T: 全attention
    '''
    def __init__(self, block_list, n_size, num_classes, genotype, dropout=0.):
        super(InsertResNet, self).__init__()
        self.inplane = 128 # 16 for cifar10 64 for cifar100 
        self.genotype = genotype
        self.dropout = dropout
        self.channel_in = self.inplane
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self._step = 4
        self.layer1 = self._make_layer(
            block_list[0],
            self.channel_in,
            blocks=n_size,
            stride=1,
            step=self._step,
            genotype=self.genotype,
        )
        self.layer2 = self._make_layer(
            block_list[1],
            self.channel_in * 2,
            blocks=n_size,
            stride=2,
            step=self._step,
            genotype=self.genotype,
        )
        self.layer3 = self._make_layer(
            block_list[2],
            self.channel_in * 4,
            blocks=n_size,
            stride=2,
            step=self._step,
            genotype=self.genotype,
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

    def _make_layer(self, block, planes, blocks, stride, step, genotype):
        strides = [stride] + [1] * (blocks - 1)
        self.layers = nn.ModuleList()
        for stride in strides:
            Block = block(self.inplane, planes, stride, step, genotype)
            self.layers += [Block]
            self.inplane = planes
        return self.layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for i, layer in enumerate(self.layer1):
            x = layer(x)
        for i, layer in enumerate(self.layer2):
            x = layer(x)
        for i, layer in enumerate(self.layer3):
            x = layer(x)
        
        x = self.avgpool(x)

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class CifarAttentionResNet(nn.Module):
    def __init__(self, block, n_size, num_classes, genotype, dropout=0.):
        super(CifarAttentionResNet, self).__init__()
        self.inplane = 64 # 16 for cifar10 64 for cifar100 
        self.genotype = genotype
        self.dropout = dropout
        self.channel_in = 64
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self._step = 4
        self.layer1 = self._make_layer(
            block,
            self.channel_in,
            blocks=n_size,
            stride=1,
            step=self._step,
            genotype=self.genotype,
        )
        self.layer2 = self._make_layer(
            block,
            self.channel_in * 2,
            blocks=n_size,
            stride=2,
            step=self._step,
            genotype=self.genotype,
        )
        self.layer3 = self._make_layer(
            block,
            self.channel_in * 4,
            blocks=n_size,
            stride=2,
            step=self._step,
            genotype=self.genotype,
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

    def _make_layer(self, block, planes, blocks, stride, step, genotype):
        strides = [stride] + [1] * (blocks - 1)
        self.layers = nn.ModuleList()
        for stride in strides:
            Block = block(self.inplane, planes, stride, step, genotype)
            self.layers += [Block]
            self.inplane = planes
        return self.layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for i, layer in enumerate(self.layer1):
            x = layer(x)
        for i, layer in enumerate(self.layer2):
            x = layer(x)
        for i, layer in enumerate(self.layer3):
            x = layer(x)

        x = self.avgpool(x)

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20(**kwargs):
    model = CifarAttentionResNet(BasicBlock, 3, **kwargs)
    return model


def resnet32(**kwargs):
    model = CifarAttentionResNet(BasicBlock, 5, **kwargs)
    return model


def resnet44(**kwargs):
    model = CifarAttentionResNet(BasicBlock, 7, **kwargs)
    return model


def resnet56(**kwargs):
    model = CifarAttentionResNet(BasicBlock, 9, **kwargs)
    return model


def rf_resnet20(**kwargs):
    model = CifarAttentionResNet(CifarRFBasicBlock, 3, **kwargs)
    return model

def rf_resnet20_ccc(**kwargs):
    model = InsertResNet([BasicBlock,BasicBlock,BasicBlock], 3, **kwargs)
    return model

def rf_resnet20_cct(**kwargs):
    model = InsertResNet([BasicBlock,BasicBlock,CifarRFBasicBlock], 3, **kwargs)
    return model

def rf_resnet20_ctt(**kwargs):
    model = InsertResNet([BasicBlock,CifarRFBasicBlock,CifarRFBasicBlock], 3, **kwargs)
    return model

def rf_resnet20_ttt(**kwargs):
    model = InsertResNet([CifarRFBasicBlock,CifarRFBasicBlock,CifarRFBasicBlock], 3, **kwargs)
    return model


def rfsa_resnet20(**kwargs):
    model = CifarAttentionResNet(CifarRFSABasicBlock, 3, **kwargs)
    return model

def rfconvnext_resnet20(**kwargs):
    model = CifarAttentionResNet(CifarRFConvNeXtBasicBlock, 3, **kwargs)
    return model

def rf_resnet32(**kwargs):
    model = CifarAttentionResNet(CifarRFBasicBlock, 5, **kwargs)
    return model

def rf_resnet56(**kwargs):
    model = CifarAttentionResNet(CifarRFBasicBlock, 9, **kwargs)
    return model

def la_resnet20(**kwargs):
    """Constructs a ResNet-20 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 3, **kwargs)
    return model

# normal resnet20 

def resnet20_cbam(**kwargs):
    model = CifarAttentionResNet(
        CifarAttentionBasicBlock, 3, cbam=True, **kwargs)
    return model


def resnet20_spp(**kwargs):
    model = CifarAttentionResNet(
        CifarAttentionBasicBlock, 3, spp=True, **kwargs)
    return model


def resnet20_se(**kwargs):
    model = CifarAttentionResNet(
        CifarAttentionBasicBlock, 3, se=True, **kwargs)
    return model

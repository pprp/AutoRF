import torch.nn as nn
import torch.nn.functional as F

from .components import CifarRFBasicBlock, CifarRFSABasicBlock


class CifarAttentionResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, PRIMITIVES=None):
        super(CifarAttentionResNet, self).__init__()
        self._steps = 4
        self.inplane = 16
        self.channel_in = 16
        self.conv1 = nn.Conv2d(3,
                               self.inplane,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(
            block,
            self.channel_in,
            blocks=n_size,
            stride=1,
            step=self._steps,
            PRIMITIVES=PRIMITIVES,
        )
        self.layer2 = self._make_layer(
            block,
            self.channel_in * 2,
            blocks=n_size,
            stride=2,
            step=self._steps,
            PRIMITIVES=PRIMITIVES,
        )
        self.layer3 = self._make_layer(
            block,
            self.channel_in * 4,
            blocks=n_size,
            stride=2,
            step=self._steps,
            PRIMITIVES=PRIMITIVES,
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

    def _make_layer(self, block, planes, blocks, stride, step, PRIMITIVES):
        strides = [stride] + [1] * (blocks - 1)
        self.layers = nn.ModuleList()
        for stride in strides:
            Block = block(self.inplane, planes, stride, step, PRIMITIVES)
            self.layers += [Block]
            self.inplane = planes
        return self.layers

    def forward(self, x, weights):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for i, layer in enumerate(self.layer1):
            x = layer(x, weights)
        for i, layer in enumerate(self.layer2):
            x = layer(x, weights)
        for i, layer in enumerate(self.layer3):
            x = layer(x, weights)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def rfsa_resnet20(**kwargs):
    model = CifarAttentionResNet(CifarRFSABasicBlock, 3, **kwargs)
    return model


def rf_resnet20(**kwargs):
    model = CifarAttentionResNet(CifarRFBasicBlock, 3, **kwargs)
    return model


def rf_resnet32(**kwargs):
    model = CifarAttentionResNet(CifarRFBasicBlock, 5, **kwargs)
    return model


def rf_resnet44(**kwargs):
    model = CifarAttentionResNet(CifarRFBasicBlock, 7, **kwargs)
    return model


def rf_resnet56(**kwargs):
    model = CifarAttentionResNet(CifarRFBasicBlock, 9, **kwargs)
    return model


def rf_resnet110(**kwargs):
    model = CifarAttentionResNet(CifarRFBasicBlock, 18, **kwargs)
    return model

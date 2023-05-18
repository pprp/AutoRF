import os
import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from torchvision.models import ResNet

from search.utils import DropPath
from space.genotypes import *
from space.operations import *
from space.spaces import OPS

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MixedOp(nn.Module):
    def __init__(self, C, stride, affine=False, PRIMITIVES=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, affine)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=affine))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class CMlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ReceptiveFieldSelfAttention(nn.Module):
    def __init__(self,
                 C,
                 steps=3,
                 reduction=False,
                 se=True,
                 drop_prob=0.,
                 mlp_ratio=2.,
                 PRIMITIVES=None):
        super(ReceptiveFieldSelfAttention, self).__init__()
        self._ops = nn.ModuleList()
        self._C = C
        self._steps = steps
        self._stride = 1
        self._se = se
        self.conv3x3 = False

        self.pos_embed = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.norm1 = nn.BatchNorm2d(C)

        for i in range(self._steps):
            for j in range(i + 1):
                op = MixedOp(C // 4,
                             self._stride,
                             False,
                             PRIMITIVES=PRIMITIVES)
                self._ops.append(op)

        self.bottle = nn.Conv2d(C,
                                C // 4,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)

        self.conv1x1 = nn.Conv2d(C // 4 * self._steps,
                                 C,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)

        self.drop_path = DropPath(
            drop_prob) if drop_prob > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(C)

        mlp_hidden_dim = int(C * mlp_ratio)

        self.mlp = CMlp(in_features=C,
                        hidden_features=mlp_hidden_dim,
                        act_layer=nn.GELU,
                        drop=0.)

        if self.conv3x3:
            self.conv3x3 = nn.Conv2d(C // 4 * self._steps,
                                     C,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False)

        if self._se:
            self.se = SE(C, reduction=4)

    def forward(self, x, weights):
        # t = self.bottle(x)
        t = x + self.pos_embed(x)
        t = self.bottle(t)

        states = [t]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # concate all released nodes
        node_out = torch.cat(states[-self._steps:], dim=1)

        if self.conv3x3:
            node_out = self.conv3x3(node_out)
        else:
            node_out = self.conv1x1(node_out)

        # shortcut
        node_out = node_out + x

        if self._se:
            node_out = self.se(node_out)

        # mlp part
        node_out = node_out + self.drop_path(self.mlp(self.norm2(node_out)))

        return node_out


class ReceptiveFieldAttention(nn.Module):
    def __init__(self, C, steps=3, reduction=False, se=True, PRIMITIVES=None):
        super(ReceptiveFieldAttention, self).__init__()
        self._ops = nn.ModuleList()
        self._C = C
        self._steps = steps
        self._stride = 1
        self._se = se
        self.conv3x3 = False

        for i in range(self._steps):
            for j in range(i + 1):
                op = MixedOp(C // 4,
                             self._stride,
                             False,
                             PRIMITIVES=PRIMITIVES)
                self._ops.append(op)

        self.bottle = nn.Conv2d(C,
                                C // 4,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)

        self.conv1x1 = nn.Conv2d(C // 4 * self._steps,
                                 C,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)

        if self.conv3x3:
            self.conv3x3 = nn.Conv2d(C // 4 * self._steps,
                                     C,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False)

        if self._se:
            self.se = SE(C, reduction=4)

    def forward(self, x, weights):
        t = self.bottle(x)

        states = [t]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # concate all released nodes
        node_out = torch.cat(states[-self._steps:], dim=1)

        if self.conv3x3:
            node_out = self.conv3x3(node_out)
        else:
            node_out = self.conv1x1(node_out)
        # shortcut
        node_out = node_out + x

        if self._se:
            node_out = self.se(node_out)

        return node_out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class CifarRFSABasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step, PRIMITIVES=None):
        super(CifarRFSABasicBlock, self).__init__()
        self._steps = step
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = ReceptiveFieldSelfAttention(planes,
                                                     PRIMITIVES=PRIMITIVES)

    def forward(self, x, weights):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out, weights)
        out = out + residual
        out = self.relu(out)

        return out


class CifarRFBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step, PRIMITIVES=None):
        super(CifarRFBasicBlock, self).__init__()
        self._steps = step
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = ReceptiveFieldAttention(planes, PRIMITIVES=PRIMITIVES)

    def forward(self, x, weights):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out, weights)
        out = out + residual
        out = self.relu(out)

        return out


class CifarAttentionResNet34(nn.Module):
    def __init__(self, block, n_size, num_classes=10):
        super(CifarAttentionResNet34, self).__init__()
        self._step = 4
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
        self.layer1 = self._make_layer(block,
                                       self.channel_in,
                                       blocks=3,
                                       stride=1,
                                       step=self._step)
        self.layer2 = self._make_layer(block,
                                       self.channel_in * 2,
                                       blocks=4,
                                       stride=2,
                                       step=self._step)
        self.layer3 = self._make_layer(block,
                                       self.channel_in * 4,
                                       blocks=6,
                                       stride=2,
                                       step=self._step)
        self.layer4 = self._make_layer(block,
                                       self.channel_in * 8,
                                       blocks=3,
                                       stride=2,
                                       step=self._step)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.channel_in * 8, num_classes)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, step):
        strides = [stride] + [1] * (blocks - 1)
        self.layers = nn.ModuleList()
        for stride in strides:
            Block = block(self.inplane, planes, stride, step)
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
        for i, layer in enumerate(self.layer4):
            x = layer(x, weights)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

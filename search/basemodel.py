import sys
import os 

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from space.genotypes import *
from space.operations import *
from space.spaces import *
from torchvision.models import ResNet

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
    def __init__(self, C, stride, affine=False):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, affine)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=affine))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class ReceptiveFieldAttention(nn.Module):
    def __init__(self, C, steps=3, reduction=False, se=False):
        super(ReceptiveFieldAttention, self).__init__()
        self._ops = nn.ModuleList()
        self._C = C
        self._steps = steps
        self._stride = 1
        self._se = se 
        for i in range(self._steps):
            for j in range(i + 1):
                op = MixedOp(C, self._stride, False)
                self._ops.append(op)
        self.conv1x1 = nn.Conv2d(C * self._steps, C, kernel_size=1, stride=1,padding=0, bias=False)
        if self._se:
            self.se = SE(self.C_in, reduction=4)

    def forward(self, x, weights):
        states = [x]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        # concate all released nodes
        node_out = torch.cat(states[-self._steps :], dim=1)
        node_out = self.conv1x1(node_out)
        # shortcut
        node_out = node_out + x
        if self._se:
            node_out = self.se(node_out)

        return node_out


class Attention(nn.Module):
    def __init__(self, steps, C):
        super(Attention, self).__init__()
        self._steps = 4
        self._C = C
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        self.C_in = self._C // 4
        self.C_out = self._C
        self.width = 4
        for i in range(self._steps):
            for j in range(1 + i):
                stride = 1
                op = MixedOp(self.C_in, stride)
                self._ops.append(op)
        self.channel_back = nn.Sequential(
            nn.Conv2d(
                self.C_in * 5, self._C, kernel_size=1, padding=0, groups=1, bias=False
            ),
            nn.BatchNorm2d(self._C),
            nn.ReLU(inplace=False),
            nn.Conv2d(self._C, self._C, kernel_size=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(self._C),
        )
        self.se = SE(self.C_in, reduction=4)
        self.se2 = SE(self.C_in * 4, reduction=16)

    def forward(self, s0, weights):
        C = s0.shape[1]
        length = C // 4
        spx = torch.split(s0, length, 1)
        spx_sum = sum(spx)
        spx_sum = self.se(spx_sum)
        offset = 0
        states = [spx[0]]
        for i in range(self._steps):
            states[0] = spx[i]
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        node_concat = torch.cat(states[-self._steps :], dim=1)
        node_concat = torch.cat((node_concat, spx_sum), dim=1)

        attention_out = self.channel_back(node_concat) + s0
        attention_out = self.se2(attention_out)

        return attention_out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

class CifarRFBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step):
        super(CifarRFBasicBlock, self).__init__()
        self._steps = step
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = ReceptiveFieldAttention(planes)

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



class CifarAttentionBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step):
        super(CifarAttentionBasicBlock, self).__init__()
        self._steps = step
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = Attention(self._steps, planes)

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


class CifarAttentionResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10):
        super(CifarAttentionResNet, self).__init__()
        self._steps = 4
        self.inplane = 16
        self.channel_in = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(
            block, self.channel_in, blocks=n_size, stride=1, step=self._steps
        )
        self.layer2 = self._make_layer(
            block, self.channel_in * 2, blocks=n_size, stride=2, step=self._steps
        )
        self.layer3 = self._make_layer(
            block, self.channel_in * 4, blocks=n_size, stride=2, step=self._steps
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CifarAttentionResNet34(nn.Module):
    def __init__(self, block, n_size, num_classes=10):
        super(CifarAttentionResNet34, self).__init__()
        self._step = 4
        self.inplane = 16
        self.channel_in = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(
            block, self.channel_in, blocks=3, stride=1, step=self._step
        )
        self.layer2 = self._make_layer(
            block, self.channel_in * 2, blocks=4, stride=2, step=self._step
        )
        self.layer3 = self._make_layer(
            block, self.channel_in * 4, blocks=6, stride=2, step=self._step
        )
        self.layer4 = self._make_layer(
            block, self.channel_in * 8, blocks=3, stride=2, step=self._step
        )
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


def la_resnet20(**kwargs):
    """Constructs a ResNet-20 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 3, **kwargs)
    return model


def la_resnet32(**kwargs):
    """Constructs a ResNet-32 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 5, **kwargs)
    return model


def la_resnet44(**kwargs):
    """Constructs a ResNet-44 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 7, **kwargs)
    return model


def la_resnet56(**kwargs):
    """Constructs a ResNet-56 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 9, **kwargs)
    return model


def la_resnet110(**kwargs):
    """Constructs a ResNet-32 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 18, **kwargs)
    return model


def attention_resnet34(**kwargs):
    """Constructs a ResNet-56 model."""
    model = CifarAttentionResNet34(CifarAttentionBasicBlock, 5, **kwargs)
    return model


def attention_resnet56(**kwargs):
    """Constructs a ResNet-34 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 9, **kwargs)
    return model


def attention_resnet110(**kwargs):
    """Constructs a ResNet-110 model."""
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 18, **kwargs)
    return model


if __name__ == "__main__":
    # m = ReceptiveFieldAttention(16)
    m = CifarRFBasicBlock(16, 32, stride=1, step=4)

    input = torch.zeros(4, 16, 32, 32)
    k = sum(1 for i in range(4) for n in range(1 + i))

    weights = torch.randn(k, len(PRIMITIVES))
    print(m)

    output = m(input, weights)

    print(output.shape)

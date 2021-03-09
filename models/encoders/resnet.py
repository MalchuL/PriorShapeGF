import torch
import torch.nn as nn

from  .Identity import Identity


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1,
                     padding=0, groups=groups, bias=False)



class BasicBlockReLU(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=True):
        super(BasicBlockReLU, self).__init__()
        if norm_layer:
            norm_layer = nn.BatchNorm1d
        else:
            norm_layer = Identity
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU()
        self.downsample = downsample

        self.inplanes = inplanes
        self.planes = planes

        if self.downsample is None and inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion),
                #nn.InstanceNorm1d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        if self.conv1.weight.device != x.device:
            raise Exception(f"{self.conv1.weight.device}, {x.device}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)

        return out


class BasicBlockLeakyReLU(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=True):
        super(BasicBlockLeakyReLU, self).__init__()
        if norm_layer:
            norm_layer = nn.BatchNorm1d
        else:
            norm_layer = Identity
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.downsample = downsample

        self.inplanes = inplanes
        self.planes = planes

        if self.downsample is None and inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion),
                #nn.InstanceNorm1d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        if self.conv1.weight.device != x.device:
            raise Exception(f"{self.conv1.weight.device}, {x.device}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)

        return out


class BasicBlockGN(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=True):
        super(BasicBlockGN, self).__init__()
        if norm_layer:
            GROUPS = 16
            norm_layer = lambda x: nn.GroupNorm(min(GROUPS, x // GROUPS), x)
        else:
            norm_layer = Identity
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU()
        self.downsample = downsample

        self.inplanes = inplanes
        self.planes = planes

        if self.downsample is None and inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion),
                #nn.InstanceNorm1d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        if self.conv1.weight.device != x.device:
            raise Exception(f"{self.conv1.weight.device}, {x.device}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)

        return out
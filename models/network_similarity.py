# -*- coding: utf-8 -*-
'''
This code is borrowed from pytorch
'''

import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, List, Type, Optional

# --------------------------------------------
# ResNet
# --------------------------------------------

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        net_type: str = 'residual'
    ) -> None:
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9, eps=1e-04, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9, eps=1e-04, affine=True)
        self.downsample = downsample
        self.stride = stride
        self.net_type = net_type

    def forward(self, x: Tensor) -> Tensor:
        if self.net_type == 'residual':
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.net_type == 'residual':
            out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        in_channels,
        block: Type[Union[BasicBlock]],
        layers: List[int],
        num_classes: int = 2,
        zero_init_residual: bool = False,
        net_type: str = 'residual',
        groups: int = 1,
        width_per_group: int = 64
    ) -> None:
        super(ResNet, self).__init__()

        self.groups = groups
        self.base_width = width_per_group
        
        self.in_channels = in_channels
        self.inplanes = 64
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, momentum=0.9, eps=1e-04, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(net_type, block, 64, layers[0], dilate=False)
        self.layer2 = self._make_layer(net_type, block, 128, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(net_type, block, 256, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(net_type, block, 512, layers[3], stride=2, dilate=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, net_type: str, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if net_type == 'residual':
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion, momentum=0.9, eps=1e-04, affine=True),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, net_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, net_type=net_type))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
def Res_Net(arch: str, num_classes: int=2):
    if arch == 'resnet18':
        model = ResNet(in_channels=1, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, 
                       zero_init_residual=True, net_type='residual')
    elif arch == 'resnet18_plain':
        model = ResNet(in_channels=1, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, 
                       zero_init_residual=False, net_type='plain')
    return model



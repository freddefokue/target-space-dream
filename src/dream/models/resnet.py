"""The stateless ResNet-20 variant used in the CIFAR-100 pilot."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StatelessLayerNorm2d(nn.GroupNorm):
    """LayerNorm-like normalization without running batch statistics."""

    def __init__(self, channels: int) -> None:
        super().__init__(1, channels)


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.norm1 = StatelessLayerNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.norm2 = StatelessLayerNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_planes, planes, 1, stride, bias=False)
            if stride != 1 or in_planes != planes
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
        return out + shortcut


class ResNet20GN1(nn.Module):
    """Pre-activation ResNet-20 with GroupNorm(1, C).

    This is the Stanton-inspired model used in the thesis experiments. It is
    not claimed to reproduce the unreleased LayerNorm implementation exactly.
    """

    def __init__(self, classes: int = 100) -> None:
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.layer1 = self._layer(16, 3, 1)
        self.layer2 = self._layer(32, 3, 2)
        self.layer3 = self._layer(64, 3, 2)
        self.norm = StatelessLayerNorm2d(64)
        self.linear = nn.Linear(64, classes)

    def _layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, current_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.norm(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


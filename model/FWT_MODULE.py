# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Dict
import torch
import torch.nn as nn

from torch import Tensor


class FWT_MODULE(torch.nn.Module):
    instance: Dict[int, Optional[FWT_MODULE]] = {}
    
    @staticmethod
    def build(index: int):
        if index not in FWT_MODULE.instance:
            FWT_MODULE.instance[index] = FWT_MODULE(16)
        return FWT_MODULE.instance[index]
    
    def __init__(self, planes: int):
        super(FWT_MODULE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learned_filter = nn.Parameter(torch.rand((planes, planes, 2, 2), device=self.device), requires_grad=True)
    
    def forward(self, vimg):
        return torch.nn.functional.conv2d(vimg, self.learned_filter, stride=2)


class DELAYED_FWT_MODULE(torch.nn.Module):
    instance: Dict[int, Optional[FWT_MODULE]] = {}
    constant_filter: Tensor
    
    @staticmethod
    def build(index: int):
        if index not in FWT_MODULE.instance:
            FWT_MODULE.instance[index] = FWT_MODULE(16)
        return FWT_MODULE.instance[index]
    
    def __init__(self, planes: int):
        super().__init__()
        temp: Tensor = torch.zeros(planes, planes, 2, 2)
        for i in range(planes): temp[i, i] = 0.25*torch.ones(2, 2)
        self.register_buffer("constant_filter", temp)
        self.learned_filter = nn.Parameter(self.constant_filter.clone(), requires_grad=True)
    
    def forward(self, vimg, use_constant: bool):
        if use_constant: weight = self.constant_filter
        else: weight = self.learned_filter
        return torch.nn.functional.conv2d(vimg, weight, stride=2)


class RESIDUAL_FWT_MODULE(torch.nn.Module):
    instance: Dict[int, Optional[FWT_MODULE]] = {}
    constant_filter: Tensor
    
    @staticmethod
    def build(index: int):
        if index not in FWT_MODULE.instance:
            FWT_MODULE.instance[index] = FWT_MODULE(16)
        return FWT_MODULE.instance[index]
    
    def __init__(self, planes: int):
        super().__init__()
        self.learned_filter = nn.Parameter(torch.randn(planes, planes, 2, 2), requires_grad=True)
        self.average_pool = nn.AvgPool2d(2)
    
    def forward(self, vimg):
        return torch.nn.functional.conv2d(vimg, self.learned_filter, stride=2) + self.average_pool(vimg)


class GROUPED_FWT_MODULE(torch.nn.Module):
    instance: Dict[int, Optional[FWT_MODULE]] = {}
    constant_filter: Tensor
    
    @staticmethod
    def build(index: int):
        if index not in FWT_MODULE.instance:
            FWT_MODULE.instance[index] = FWT_MODULE(16)
        return FWT_MODULE.instance[index]
    
    def __init__(self, planes: int):
        super().__init__()
        self.convolution = nn.Conv2d(planes, planes, kernel_size=2, stride=2, groups=planes)
        self.average_pool = nn.AvgPool2d(2)
    
    def forward(self, vimg):
        return self.convolution(vimg) + self.average_pool(vimg)

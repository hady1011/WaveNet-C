# -*- coding: utf-8 -*-
from typing import List

import torch
import torch.nn as nn


#you can use this formula output size = [(W−K+2P)/S]+1.

#W is the input volume - in your case 128
#K is the Kernel size - in your case 5
#P is the padding - in your case 0 i believe
#S is the stride - which you have not provided.
from torch import Tensor

from model.FWT_MODULE import FWT_MODULE


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class WaveletAttention_WAV(torch.nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']
    return_indices: bool
    ceil_mode: bool
    def __init__(self, planes, type):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # y = torch.tensor([0])
        # self.register_buffer("y", torch.tensor([0]))
        
        if(type == 'Basic'):
            self._excitation = nn.Sequential(
                nn.Linear(in_features=planes, out_features=round(planes / 2), device=self.device),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=round(planes / 2), out_features=planes, device=self.device),
                nn.Sigmoid(),
            )
        if(type == 'Bottle'):
            self._excitation = nn.Sequential(
                nn.Linear(in_features=planes * 4, out_features=round(planes / 4), device=self.device),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=round(planes / 4), out_features=planes * 4, device=self.device),
                nn.Sigmoid(),
            )
        
    def forward(self, FWT: List[FWT_MODULE], input: List[Tensor]):
        while input[0].size(-1) > 1:
            input = [f(x) for f, x in zip(FWT, input)]
        temp = torch.cat(input, dim=1)
        b, c = temp.size()
        attention = self._excitation(temp.view(b, -1)).view(b, c, 1, 1)
        out = torch.cat(input, dim=1) * attention
        return out
    
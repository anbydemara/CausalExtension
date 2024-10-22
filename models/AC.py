#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：AC.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/10/22 15:28 
"""
from torch import nn


class InsEncoder(nn.Module):
    def __init__(self):
        super(InsEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # Encoder
            # input (b, 512)
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512)
            # input (b, 64)
        )

    def forward(self, *input):
        out = self.encoder(*input)
        return out


class InsDecoder(nn.Module):
    def __init__(self):
        super(InsDecoder, self).__init__()

        self.decoder = nn.Sequential(
        # DEcoder
        nn.Linear(512, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 2048)
        )

    def forward(self, *input):
        out = self.decoder(*input)
        return out
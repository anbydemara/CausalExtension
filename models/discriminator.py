#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：discriminator.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/10/7 15:03 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes, patch_size):
        super(Discriminator, self).__init__()
        dim = 512
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'):

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj

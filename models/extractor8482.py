#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：extractor8482.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/10/13 10:49 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.ResNet import get_resnet8
# from models.Res3D import ResNet3D

# class Extractor(nn.Module):
#     """
#     Module name: feature extractor
#     Description: extracting the causal features
#     version: out dim=256
#     """
#     def __init__(self, in_channels, zdim):
#         super(Extractor, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3)
#         self.conv2 = nn.Conv2d(128, 128, kernel_size=3)
#         self.maxpool = nn.MaxPool2d(3, 3)
#
#         self.fc1 = nn.Linear(in_features=128, out_features=zdim)
#         self.fc2 = nn.Linear(in_features=zdim, out_features=zdim)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.maxpool(x)
#         x = F.relu(self.conv2(x))
#         x = self.maxpool(x)
#         x = x.view(x.size(0), -1)   # 保留批处理维度
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x

class Extractor(nn.Module):
    """
    Module name: feature extractor
    Description: extracting the causal features
    up: change out dim to 2048
    """
    def __init__(self, in_channels, zdim):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3)
        self.maxpool = nn.MaxPool2d(3, 3)

        self.fc1 = nn.Linear(in_features=128, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=zdim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)   # 保留批处理维度
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x    # 256

class Classifier(nn.Module):
    """
    Module name: classifier
    Description: classifier model by MLP
    Other: another solution -- KAN
    """
    def __init__(self, input_size, num_classes, hidden_size=128):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # return F.softmax(x, dim=1)
        return F.normalize(x)


class CausalNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CausalNet, self).__init__()
        self.swl = SWL(in_channels=in_channels)
        self.extractor = Extractor(in_channels, 256)
        # self.extractor = get_resnet8(102)
        # self.extractor = ResNet3D(1, 8, 16, n_bands=102, patch_size=17, embed_dim=512, CLASS_NUM=num_classes)
        # self.classifier = Classifier(input_size=256, num_classes=num_classes, hidden_size=128)
        self.classifier = Classifier(input_size=512, num_classes=num_classes, hidden_size=128)

    def forward(self, x):
        band_weights = self.swl(x)
        x = x + x * band_weights
        features = self.extractor(x)
        return self.classifier(features), band_weights.squeeze(), features


class SWL(nn.Module):
    def __init__(self, in_channels=102):
        super(SWL, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels // 8, out_channels=in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        B, C, _, _ = x.size()
        x_avg, x_max = self.avgpool(x), self.maxpool(x)
        avg_weights, max_weights = self.channel_attention(x_avg), self.channel_attention(x_max)
        band_weights = self.sigmoid(avg_weights + max_weights)
        return band_weights


class CategoryConsistencyLoss(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(CategoryConsistencyLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.weightcenters = nn.Parameter(torch.normal(0, 1, (num_classes, embedding_size)))

    def forward(self, x, labels):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        dist_metric = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                      torch.pow(self.weightcenters, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        dist_metric.addmm_(x, self.weightcenters.t(), beta=1, alpha=-2)

        dist = dist_metric[range(batch_size), labels]
        loss = dist.clamp(1e-12, 1e+12).sum() / batch_size
        return loss
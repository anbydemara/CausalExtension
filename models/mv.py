#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：mv.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/10/19 10:55 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义多视角特征提取模块
class MultiViewFeatureExtractor(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(MultiViewFeatureExtractor, self).__init__()
        # 定义三个视角的卷积网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 三个视角的特征提取
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        return feat1, feat2, feat3


# 定义域判别器
class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)  # 二分类：源域 vs 目标域
        )

    def forward(self, x):
        return self.fc(x)


# 定义完整的跨场景分类网络
class CrossSceneClassifier(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_classes):
        super(CrossSceneClassifier, self).__init__()
        # 多视角特征提取器
        self.feature_extractor = MultiViewFeatureExtractor(input_channels, hidden_dim)

        # 分类器，用于最终的分类任务
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )

        # 域判别器
        self.domain_discriminator = DomainDiscriminator(hidden_dim * 3)

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # 提取多视角特征
        feat1, feat2, feat3 = self.feature_extractor(x)

        # 融合多视角特征
        fused_feat = torch.cat([feat1, feat2, feat3], dim=1)

        # 进行分类任务
        class_output = self.classifier(fused_feat.view(fused_feat.size(0), -1))

        # 域分类输出
        domain_output = self.domain_discriminator(fused_feat.view(fused_feat.size(0), -1))

        return class_output, domain_output, feat1, feat2, feat3

    def compute_multi_view_loss(self, feat1, feat2, feat3):
        # 计算多视角特征之间的MSE损失，确保视角一致性
        dif12 = self.mse_loss(feat1, feat2.detach())
        dif13 = self.mse_loss(feat1, feat3.detach())
        dif23 = self.mse_loss(feat2, feat3.detach())
        return (dif12 + dif13 + dif23) / 3

    def compute_domain_adversarial_loss(self, domain_output, domain_labels):
        # 域对抗损失（通过反转梯度来增强域对抗效果）
        return self.ce_loss(domain_output, domain_labels)
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：MVDC.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/10/14 9:52 
"""
import torch
import torch.nn as nn

from models.generator import Generator


class AdaptiveEncoder(nn.Module):
    def __init__(self, input_channels):
        super(AdaptiveEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 6 * 6, 256)  # 假设输入大小为 13*13

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平层
        x = torch.relu(self.fc(x))
        return x


class CausalInferenceModule(nn.Module):
    def __init__(self, lambda_causal=0.1):
        super(CausalInferenceModule, self).__init__()
        self.lambda_causal = lambda_causal

    def forward(self, features, domain_predictions):
        # 随机扰动输入特征 (假设干预)
        intervention = features + torch.randn_like(features) * 0.1
        domain_predictions_intervention = domain_predictions(intervention)

        # 因果损失：假设扰动后，域分类结果应该保持不变
        causal_loss = nn.MSELoss()(domain_predictions, domain_predictions_intervention)
        return causal_loss * self.lambda_causal


class CausalMVDC(nn.Module):
    def __init__(self, num_views=3, input_dim=256, hidden_dim=128, num_domains=2, lambda_causal=0.1):
        super(CausalMVDC, self).__init__()
        self.num_views = num_views
        self.encoders = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) for _ in range(num_views)])

        self.domain_classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, num_domains),
            nn.Softmax(dim=1)
        ) for _ in range(num_views)])

        self.causal_module = CausalInferenceModule(lambda_causal)

    def forward(self, features):
        domain_outputs = []
        causal_losses = 0.0

        for i in range(self.num_views):
            encoded = self.encoders[i](features)
            domain_pred = self.domain_classifiers[i](encoded)
            domain_outputs.append(domain_pred)

            # 计算因果损失
            causal_losses += self.causal_module(encoded, domain_pred)

        return domain_outputs, causal_losses


class CausalDomainGeneralizationModel(nn.Module):
    def __init__(self, input_channels, num_views=3, num_domains=2, lambda_causal=0.1):
        super(CausalDomainGeneralizationModel, self).__init__()
        self.encoder = AdaptiveEncoder(input_channels)
        self.mvdc = CausalMVDC(num_views=num_views, input_dim=256, hidden_dim=128, num_domains=num_domains,
                               lambda_causal=lambda_causal)

    def forward(self, x):
        # 提取特征
        features = self.encoder(x)

        # 多视角域分类和因果推断
        domain_outputs, causal_losses = self.mvdc(features)

        return domain_outputs, causal_losses

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))  # 输出值介于 0 和 1 之间，表示真实或伪域
        return x

class CausalDomainGeneralizationWithGAN(nn.Module):
    def __init__(self, input_channels, num_views=3, num_domains=2, lambda_causal=0.1):
        super(CausalDomainGeneralizationWithGAN, self).__init__()
        self.encoder = AdaptiveEncoder(input_channels)
        self.mvdc = CausalMVDC(num_views=num_views, input_dim=256, hidden_dim=128, num_domains=num_domains,
                               lambda_causal=lambda_causal)
        self.generator = Generator(input_channels, input_channels)  # 生成与输入同维度的伪域数据
        self.discriminator = Discriminator(input_channels)

    def forward(self, x):
        # 生成伪域数据
        fake_data = self.generator(x)

        # 真实数据和伪域数据的判别输出
        real_output = self.discriminator(x)
        fake_output = self.discriminator(fake_data)

        # 提取真实域和伪域的特征
        features_real = self.encoder(x)
        features_fake = self.encoder(fake_data)

        # 多视角域分类和因果推断 (使用真实域特征)
        domain_outputs_real, causal_losses_real = self.mvdc(features_real)

        # 返回多视角输出，因果损失，判别输出和生成的数据
        return domain_outputs_real, causal_losses_real, real_output, fake_output, fake_data

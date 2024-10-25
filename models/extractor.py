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

from models.functions import ReverseLayerF


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
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=zdim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # 保留批处理维度
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


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
        return x


#         return F.softmax(x, dim=1)
#         return F.normalize(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_size):
        super(DomainClassifier, self).__init__()

        self.d_fc1 = nn.Linear(in_features=input_size, out_features=100)
        self.d_bn1 = nn.BatchNorm1d(100)
        self.d_relu1 = nn.ReLU(True)
        self.d_fc2 = nn.Linear(in_features=100, out_features=2)
        self.d_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.d_bn1(self.d_fc1(x))


class CausalNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(CausalNet, self).__init__()
        #         self.swl = SWL(in_channels=in_channels)
        self.extractor = Extractor(in_channels, out_channels)
        #         self.classifier = nn.Sequential(
        #             nn.Linear(out_channels, 100),
        #             nn.BatchNorm1d(100),
        #             nn.ReLU(True),
        #             nn.Dropout(),
        #             nn.Linear(100, 100),
        #             nn.BatchNorm1d(100),
        #             nn.ReLU(True),
        #             nn.Linear(100, num_classes),
        #             nn.LogSoftmax(dim=1)
        #         )
        self.classifier = Classifier(input_size=out_channels, num_classes=num_classes)

        # 单视角域分类器
        self.MV_domainclassifier = MVDClassifier()

        # 多视角En-Decoder
        self.mv1_encoder = DomainEncoder()
        self.mv2_encoder = DomainEncoder()
        self.mv3_encoder = DomainEncoder()
        self.mv1_decoder = DomainDecoder()
        self.mv2_decoder = DomainDecoder()
        self.mv3_decoder = DomainDecoder()

        # 多视角域分类器
        self.MV1_classifier = MVDC_En()
        self.MV2_classifier = MVDC_En()
        self.MV3_classifier = MVDC_En()

        self.ln_ins = nn.LayerNorm(normalized_shape = [256])

        self.domain_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        # self.consistency_loss = nn.MSELoss(size_average=False)

        # self.domainclassifier = nn.Sequential(
        #     nn.Linear(in_features=out_channels, out_features=100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(),
        #     nn.Linear(in_features=100, out_features=2),
        #     nn.LogSoftmax(dim=1)
        # )

    def forward(self, x, domain_label, alpha, mode='test'):
        features = self.extractor(x)
        if mode == 'test':
            return self.classifier(features)
        elif mode == 'train':
            # 单视角
            sigmoid_out = self.MV_domainclassifier(features, alpha)

            # 多视角
            # 视角1
            MV1_feat = self.mv1_encoder(features)
            re1_feat = self.mv1_decoder(MV1_feat)

            #    重构损失
            re1_loss = self.mse_loss(re1_feat, features)
            #    归一化
            MV1_feat = self.ln_ins(MV1_feat)

            #    域分类损失
            mv1_sigmoid_out = self.MV1_classifier(MV1_feat, alpha)
            mv1_loss = self.domain_loss(mv1_sigmoid_out, domain_label)

            # 视角2
            MV2_feat = self.mv2_encoder(features)
            re2_feat = self.mv2_decoder(MV2_feat)

            #    重构损失
            re2_loss = self.mse_loss(re2_feat, features)
            #    归一化
            MV2_feat = self.ln_ins(MV2_feat)

            #    域分类损失
            mv2_sigmoid_out = self.MV2_classifier(MV2_feat, alpha)
            mv2_loss = self.domain_loss(mv2_sigmoid_out, domain_label)

            # 视角3
            MV3_feat = self.mv3_encoder(features)
            re3_feat = self.mv3_decoder(MV3_feat)

            #    重构损失
            re3_loss = self.mse_loss(re3_feat, features)
            #    归一化
            MV3_feat = self.ln_ins(MV3_feat)

            #    域分类损失
            mv3_sigmoid_out = self.MV3_classifier(MV3_feat, alpha)
            mv3_loss = self.domain_loss(mv3_sigmoid_out, domain_label)

            # 多视角损失
            dif12_ins = (self.mse_loss(MV1_feat, MV2_feat.detach()) + self.mse_loss(MV2_feat, MV1_feat.detach()))/2
            dif13_ins = (self.mse_loss(MV1_feat, MV3_feat.detach()) + self.mse_loss(MV3_feat, MV1_feat.detach()))/2
            dif23_ins = (self.mse_loss(MV2_feat, MV3_feat.detach()) + self.mse_loss(MV3_feat, MV2_feat.detach()))/2

            # 一致性损失
            # self.consistency_loss(sigmoid_out)

            # 汇总
            re_loss = re1_loss + re2_loss + re3_loss
            mv_loss = mv1_loss + mv2_loss + mv3_loss
            dif_loss = dif12_ins + dif13_ins + dif23_ins
            return self.classifier(features), features, sigmoid_out, re_loss, mv_loss, dif_loss


class DomainEncoder(nn.Module):
    def __init__(self):
        super(DomainEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            # nn.ReLU(True),
            # nn.Linear(1024, 512)
        )

    def forward(self, *input):
        out = self.encoder(*input)
        return out


class DomainDecoder(nn.Module):
    def __init__(self):
        super(DomainDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
        )

    def forward(self, *input):
        out = self.decoder(*input)
        return out


class MVDClassifier(nn.Module):
    """
    单视角域分类器
    """

    def __init__(self):
        super(MVDClassifier, self).__init__()
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 100)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.dc_classifier = nn.Linear(100, 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x = self.sigmoid(self.dc_classifier(x))
        return x

class MVDC_En(nn.Module):
    def __init__(self):
        super(MVDC_En, self).__init__()
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(128, 100)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.dc_classifier = nn.Linear(100, 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x = self.sigmoid(self.dc_classifier(x))
        return x
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：MC.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/10/22 15:32 
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from models.AC import InsEncoder, InsDecoder
from models.DA import _InstanceDA_En


class MVDC(nn.Module):
    def __init__(self, bs, device=0):
        super(MVDC, self).__init__()
        self.batch_size = bs
        self.InsDA_en1 = _InstanceDA_En()
        self.InsDA_en2 = _InstanceDA_En()
        self.InsDA_en3 = _InstanceDA_En()

        self.InsEn_1 = InsEncoder()
        self.InsDe_1 = InsDecoder()

        self.InsEn_2 = InsEncoder()
        self.InsDe_2 = InsDecoder()

        self.InsEn_3 = InsEncoder()
        self.InsDe_3 = InsDecoder()

        self.ln_ins = nn.LayerNorm(normalized_shape=[512])

        self.InsDA_loss = nn.BCELoss()

        self.mse_loss = nn.MSELoss()

        self.device = device

    def forward(self, feat_s1, feat_s2):
        # region [视角1]
        MV1_ins_feat_s1 = self.InsEn_1(feat_s1)
        MV1_ins_feat_s2 = self.InsEn_1(feat_s2)
        re1_ins_feat_s1 = self.InsDe_1(MV1_ins_feat_s1)
        re1_ins_feat_s2 = self.InsDe_1(MV1_ins_feat_s2)
        # 归一化
        MV1_ins_feat_s1 = self.ln_ins(MV1_ins_feat_s1)
        MV1_ins_feat_s2 = self.ln_ins(MV1_ins_feat_s2)

        # 1. recon_loss(重构损失)
        MV1_ins_s1 = self.mse_loss(re1_ins_feat_s1, feat_s1.detach())
        MV1_ins_s2 = self.mse_loss(re1_ins_feat_s2, feat_s2.detach())

        # 2. 领域分类损失
        # s1
        instance_sigmoid_s1_MV1, same_size_label_s1_MV1 = self.InsDA_en1(MV1_ins_feat_s1, Variable(
            torch.FloatTensor([0.] * self.batch_size)))
        DA_ins_MV1_s1 = self.InsDA_loss(instance_sigmoid_s1_MV1, same_size_label_s1_MV1)

        # s2
        instance_sigmoid_s2_MV1, same_size_label_s2_MV1 = self.InsDA_en1(MV1_ins_feat_s2, Variable(
            torch.FloatTensor([1.] * self.batch_size)))
        DA_ins_MV1_s2 = self.InsDA_loss(instance_sigmoid_s2_MV1, same_size_label_s2_MV1)

        # ins_MV1_loss = DA_ins_MV1_s1 + DA_ins_MV1_s2 + MV1_ins_s1 + MV1_ins_s2
        ins_MV1_recon_loss = MV1_ins_s1 + MV1_ins_s2
        ins_MV1_cls_loss = DA_ins_MV1_s1 + DA_ins_MV1_s2
        # end region [视角1]

        # region [视角2]
        MV2_ins_feat_s1 = self.InsEn_2(feat_s1)
        MV2_ins_feat_s2 = self.InsEn_2(feat_s2)
        re2_ins_feat_s1 = self.InsDe_2(MV2_ins_feat_s1)
        re2_ins_feat_s2 = self.InsDe_2(MV2_ins_feat_s2)
        # 归一化
        MV2_ins_feat_s1 = self.ln_ins(MV2_ins_feat_s1)
        MV2_ins_feat_s2 = self.ln_ins(MV2_ins_feat_s2)

        # 1. recon_loss(重构损失)
        MV2_ins_s1 = self.mse_loss(re2_ins_feat_s1, feat_s1.detach())
        MV2_ins_s2 = self.mse_loss(re2_ins_feat_s2, feat_s2.detach())

        # 2. 领域分类损失
        # s1
        instance_sigmoid_s1_MV2, same_size_label_s1_MV2 = self.InsDA_en2(MV2_ins_feat_s1, Variable(
            torch.FloatTensor([0.] * self.batch_size)))
        DA_ins_MV2_s1 = self.InsDA_loss(instance_sigmoid_s1_MV2, same_size_label_s1_MV2)
        # s2
        instance_sigmoid_s2_MV2, same_size_label_s2_MV2 = self.InsDA_en2(MV2_ins_feat_s2, Variable(
            torch.FloatTensor([1.] * self.batch_size)))
        DA_ins_MV2_s2 = self.InsDA_loss(instance_sigmoid_s2_MV2, same_size_label_s2_MV2)

        # ins_MV2_loss = DA_ins_MV2_s1 + DA_ins_MV2_s2 + MV2_ins_s1 + MV2_ins_s2
        ins_MV2_recon_loss = MV2_ins_s1 + MV2_ins_s2
        ins_MV2_cls_loss = DA_ins_MV2_s1 + DA_ins_MV2_s2

        # end region [视角2]

        # region [视角3]
        MV3_ins_feat_s1 = self.InsEn_3(feat_s1)
        MV3_ins_feat_s2 = self.InsEn_3(feat_s2)
        re3_ins_feat_s1 = self.InsDe_3(MV3_ins_feat_s1)
        re3_ins_feat_s2 = self.InsDe_3(MV3_ins_feat_s2)
        # 归一化
        MV3_ins_feat_s1 = self.ln_ins(MV3_ins_feat_s1)
        MV3_ins_feat_s2 = self.ln_ins(MV3_ins_feat_s2)

        # 1. recon_loss(重构损失)
        MV3_ins_s1 = self.mse_loss(re3_ins_feat_s1, feat_s1.detach())
        MV3_ins_s2 = self.mse_loss(re3_ins_feat_s2, feat_s2.detach())

        # 2. 领域分类损失
        # s1
        instance_sigmoid_s1_MV3, same_size_label_s1_MV3 = self.InsDA_en3(MV3_ins_feat_s1, Variable(
            torch.FloatTensor([0.] * self.batch_size)))
        DA_ins_MV3_s1 = self.InsDA_loss(instance_sigmoid_s1_MV3, same_size_label_s1_MV3)
        # s2
        instance_sigmoid_s2_MV3, same_size_label_s2_MV3 = self.InsDA_en3(MV3_ins_feat_s2, Variable(
            torch.FloatTensor([1.] * self.batch_size)))
        DA_ins_MV3_s2 = self.InsDA_loss(instance_sigmoid_s2_MV3, same_size_label_s2_MV3)

        # ins_MV3_loss = DA_ins_MV3_s1 + DA_ins_MV3_s2 + MV3_ins_s1 + MV3_ins_s2
        ins_MV3_recon_loss = MV3_ins_s1 + MV3_ins_s2
        ins_MV3_cls_loss = DA_ins_MV3_s1 + DA_ins_MV3_s2
        # end region [视角2]

        # 3. 多视角损失
        dif12_ins_s1 = (self.mse_loss(MV1_ins_feat_s1, MV2_ins_feat_s1.detach()) + self.mse_loss(MV2_ins_feat_s1,
                                                                                                 MV1_ins_feat_s1.detach())) / 2
        dif12_ins_s2 = (self.mse_loss(MV1_ins_feat_s2, MV2_ins_feat_s2.detach()) + self.mse_loss(MV2_ins_feat_s2,
                                                                                                 MV1_ins_feat_s2.detach())) / 2

        dif13_ins_s1 = (self.mse_loss(MV1_ins_feat_s1, MV3_ins_feat_s1.detach()) + self.mse_loss(MV3_ins_feat_s1,
                                                                                                 MV1_ins_feat_s1.detach())) / 2
        dif13_ins_s2 = (self.mse_loss(MV1_ins_feat_s2, MV3_ins_feat_s2.detach()) + self.mse_loss(MV3_ins_feat_s2,
                                                                                                 MV1_ins_feat_s2.detach())) / 2

        dif23_ins_s1 = (self.mse_loss(MV3_ins_feat_s1, MV2_ins_feat_s1.detach()) + self.mse_loss(MV2_ins_feat_s1,
                                                                                                 MV3_ins_feat_s1.detach())) / 2
        dif23_ins_s2 = (self.mse_loss(MV3_ins_feat_s2, MV2_ins_feat_s2.detach()) + self.mse_loss(MV2_ins_feat_s2,
                                                                                                 MV3_ins_feat_s2.detach())) / 2

        # ins_mv_dis_loss = torch.exp(-(dif12_ins_s1 + dif12_ins_s2 + dif13_ins_s1 + dif13_ins_s2 + dif23_ins_s1 + dif23_ins_s2))
        ins_mv_dis_loss = 1 / (dif12_ins_s1 + dif12_ins_s2 + dif13_ins_s1 + dif13_ins_s2 + dif23_ins_s1 + dif23_ins_s2)

        # ins_MV_loss = ins_MV1_loss + ins_MV2_loss + ins_MV3_loss + un_ins_dis # - (0.01) * (dif_ins_s1 + dif_ins_s2)
        ins_mv_recon_loss = ins_MV3_recon_loss + ins_MV2_recon_loss + ins_MV1_recon_loss
        ins_mv_cls_loss = ins_MV3_cls_loss + ins_MV2_cls_loss + ins_MV1_cls_loss


        return ins_mv_dis_loss, ins_mv_recon_loss, ins_mv_cls_loss

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

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)   # 保留批处理维度
        return x

if __name__ == '__main__':
    extractor = Extractor(102, 512)
    mc = MVDC(256, 0)
    x = torch.randn((256, 102, 13, 13))
    feat1 = extractor(x)
    feat2 = extractor(x)
    ins_mv_dis_loss, ins_mv_recon_loss, ins_mv_cls_loss  = mc(feat1, feat2)

    # ins_MV_loss = 0.1 * ins_mv_recon_loss + 0.1 * ins_mv_cls_loss + 0.01 * ins_mv_dis_loss
    da_MV_loss = 0.1 * ins_mv_recon_loss + 0.1 * ins_mv_cls_loss + 0.01 * ins_mv_dis_loss
    print(da_MV_loss)
    # print(ins_mv_dis_loss.item(), ins_mv_recon_loss.item(), ins_mv_cls_loss.item())

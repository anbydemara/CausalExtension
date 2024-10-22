#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：SCG_Augmented.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/10/12 23:28 
"""
import cv2
import numpy as np
import scipy.io as sio

def load_hyperspectral_image(mat_file_path):
    mat = sio.loadmat(mat_file_path)
    img = mat['ori_data']
    return img


def FreCom_hyper(img):
    h, w, c = img.shape
    img_dct = np.zeros((h, w, c))
    for i in range(c):
        img_ = img[:, :, i]  # 获取每个波段
        img_ = np.float32(img_)  # 将数值精度调整为32位浮点型
        img_dct[:, :, i] = cv2.dct(img_)  # 使用dct获得img的频域图像

    return img_dct


def Matching_hyper(img):
    # theta = np.random.uniform(alpha, beta)
    h, w, c = img.shape
    img_dct = FreCom_hyper(img)

    mask = np.zeros((h, w, c))
    v1 = int(min(h, w) * 0.005)  # 低中频划分
    v2 = int(min(h, w) * 0.7)  # 中高频划分
    v3 = min(h, w)
    # 简便带通滤波器设计
    for x in range(h):
        for y in range(w):
            if (max(x, y) <= v1):
                mask[x][y] = 1 - max(x, y) / v1 * 0.95
            elif (v1 < max(x, y) <= v2):
                mask[x][y] = 0.01
            elif (v2 <= max(x, y) <= v3):
                mask[x][y] = (max(x, y) - v2) / (v3 - v2) * 0.3
            else:
                mask[x][y] = 0.5
    n_mask = 1 - mask
    # 划分为因果部分和非因果部分
    non_img_dct = img_dct * mask
    cal_img_dct = img_dct * n_mask

    # 非因果部分随即变换
    ref_dct = np.zeros_like(non_img_dct)
    for i in range(c):
        ref_dct[:, :, i] = non_img_dct[:, :, i] * (1 + np.random.randn())

    # 重新组合
    img_fc = ref_dct + cal_img_dct

    img_out = np.zeros((h, w, c))
    for i in range(c):
        img_out[:, :, i] = cv2.idct(img_fc[:, :, i]).clip(min_val, max_val)

    return img_out


if __name__ == '__main__':
    mat_path = './data/Pavia/PaviaU.mat'
    mat = sio.loadmat(mat_path)
    img = mat['ori_data']
    print(img.shape)
    exit(0)
    min_val = np.min(img)  # 高光谱图像的最小值
    max_val = np.max(img)  # 高光谱图像的最大值
    # # 生成一个随机参考图像（同样尺寸和波段）
    # h, w, c = img.shape
    # reference = np.ones_like(img)
    # for i in range(c):
    #     # reference[:, :, i] = reference[:, :, i] * np.random.randint(0, 255)
    #     reference[:, :, i] = reference[:, :, i] * np.random.randint(min_val, max_val)

    # 对高光谱图像进行频域匹配处理
    img_matched = Matching_hyper(img)

    img_matched = np.int16(img_matched)
    mat['ori_data'] = img_matched
    sio.savemat('./data/Pavia/PaviaU_A.mat', mat)

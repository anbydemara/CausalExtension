import torch
import pywt
import numpy as np


def apply_dwt_on_batch(batch):
    """
    对一个批次的图像进行 DWT 处理，获取低频子带（LL）
    batch: Tensor，大小为 (128, 102, 13, 13)，即批次大小为128，每张图像为102x13x13
    return: Tensor，低频子带（LL）部分，大小为 (128, 102, 7, 7)
    """
    # batch size: 128, channel: 102, height: 13, width: 13
    batch_size, channels, height, width = batch.shape

    # 创建一个空的张量来保存 DWT 后的低频部分（LL）
    ll_part = torch.zeros(batch_size, channels, 7, 7)
    print(ll_part.shape)
    for i in range(batch_size):
        for j in range(channels):
            # 对每张图像的每个通道进行 DWT 分解
            image = batch[i, j].numpy()  # 转为numpy数组

            # 使用pywt.dwt2进行二维离散小波变换
            coeffs2 = pywt.dwt2(image, 'haar')  # 'haar' 是一个常用的离散小波
            LL, (LH, HL, HH) = coeffs2

            # 仅提取低频子带（LL）
            ll_part[i, j] = torch.tensor(LL)  # 将LL转换回Tensor，并存储在结果中

    return ll_part


# 假设输入的批次大小是128，大小为102x13x13的图像
batch = torch.randn(128, 102, 13, 13)  # 随机生成一个批次

# 获取低频子带（LL）
ll_batch = apply_dwt_on_batch(batch)

print(f"Low-frequency subband (LL) batch shape: {ll_batch.shape}")

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：train_script.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/9/27 15:59 
"""
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config.config import get_args
from data.datasets import get_dataset, HyperX
from models.MVDC import CausalDomainGeneralizationWithGAN
from models.con_loss import SupConLoss
from models.discriminator import Discriminator
from models.generator import Generator
from utils.data_util import sample_gt, seed_worker, metrics


def set_seed(seed):
    """
    随机数种子设置
    """
    # torch seed
    torch.manual_seed(seed)  # cup seed
    torch.cuda.manual_seed_all(seed)  # multi-gpu seed

    # python & numpy seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化

    # cudnn seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def generator_loss(fake_output):
    # 生成器希望判别器认为生成的伪域数据是真实的
    return nn.BCELoss()(fake_output, torch.ones_like(fake_output))

def discriminator_loss(real_output, fake_output):
    # 判别器需要正确分类真实域和伪域数据
    real_loss = nn.BCELoss()(real_output, torch.ones_like(real_output))  # 真实域
    fake_loss = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))  # 伪域
    return (real_loss + fake_loss) / 2

def train_with_gan(epoch, model, train_loader, optimizer_gen, optimizer_dis, optimizer_main, domain_criterion):
    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer_main.zero_grad()
        optimizer_gen.zero_grad()
        optimizer_dis.zero_grad()

        # 前向传播，只传入图像数据
        domain_outputs_real, causal_loss, real_output, fake_output, fake_data = model(images)

        # 计算域分类损失 (多视角对抗域分类器)
        domain_loss = sum([domain_criterion(output, labels) for output in domain_outputs_real])

        # 计算判别器损失 (GAN 对抗损失)
        dis_loss = discriminator_loss(real_output, fake_output)

        # 计算生成器损失 (使生成数据尽可能接近真实数据)
        gen_loss = generator_loss(fake_output)

        # 总损失 = 域分类损失 + 因果损失 + 判别器和生成器损失
        total_main_loss = domain_loss + causal_loss
        total_main_loss.backward(retain_graph=True)
        optimizer_main.step()  # 优化主模型，包括 Encoder 和 MVDC

        # 优化生成器
        gen_loss.backward()
        optimizer_gen.step()  # 优化生成器，使其生成的伪域数据更逼真

        # 优化判别器
        dis_loss.backward()
        optimizer_dis.step()  # 优化判别器，以正确区分真实域和伪域数据

        total_loss += total_main_loss.item() + gen_loss.item() + dis_loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')


def evaluate(net, val_loader, gpu, tgt=False):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(gpu)
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max()+1)
        print(results['Confusion_matrix'],'\n','TPR:', np.round(results['TPR']*100,2),'\n', 'OA:', results['Accuracy'])
    return acc


def evaluate_tgt(cls_net, gpu, loader, modelpath):
    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['Discriminator'])
    cls_net.eval()
    teacc = evaluate(cls_net, loader, gpu, tgt=True)
    return teacc

if __name__ == '__main__':
    # 全局参数 & 设置
    DATA_ROOT = './data/datasets/'
    args = get_args()
    hyperparams = vars(args)
    set_seed(args.seed)

    ## log
    root = os.path.join(args.save_path, args.source_domain)
    sub_dir = 'lr' + str(args.lr) + '_ps' + str(args.patch_size) + '_bs' + str(
        args.batch_size) + '_' + datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    log_dir = os.path.join(str(root), sub_dir)

    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    # 数据加载
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                        os.path.join(DATA_ROOT, args.data_path))
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                        os.path.join(DATA_ROOT, args.data_path))

    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    if tmp < 1:
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = DataLoader(train_dataset,
                              batch_size=hyperparams['batch_size'],
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=g,
                              shuffle=True, )
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = DataLoader(val_dataset,
                            pin_memory=True,
                            batch_size=hyperparams['batch_size'])
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = DataLoader(test_dataset,
                             pin_memory=True,
                             worker_init_fn=seed_worker,
                             generator=g,
                             batch_size=hyperparams['batch_size'])
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    model = CausalDomainGeneralizationWithGAN(input_channels=N_BANDS)

    domain_criterion = nn.CrossEntropyLoss()
    optimizer_gen = torch.optim.Adam(model.generator.parameters(), lr=1e-4)
    optimizer_dis = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4)
    optimizer_main = torch.optim.Adam(list(model.encoder.parameters()) + list(model.mvdc.parameters()), lr=1e-4)

    best_acc = 0
    taracc, taracc_list = 0, []
    for epoch in range(1, args.epoch + 1):
        start = time.time()
        train_with_gan(epoch, model, train_loader, optimizer_gen, optimizer_dis, optimizer_main, domain_criterion)
        end = time.time()
    writer.close()
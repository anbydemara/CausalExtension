#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：train_young8482.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/10/22 15:26 
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
from tqdm import tqdm

from config.config import get_args
from data.datasets import get_dataset, HyperX
from models.extractor import CausalNet, CategoryConsistencyLoss
from models.generator import Generator
from utils.data_util import sample_gt, seed_worker, metrics
from utils.train_util import save_model, get_metrics, factorization_loss, adjust_learning_rate


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


def train(epoch):
    model.train()
    loss_list = []
    train_loss = 0  # loss
    loop = tqdm(train_loader, desc='Epoch [{}/{}]'.format(epoch, args.epoch))
    for i, (data, label) in enumerate(loop):
        data, label = data.to(args.gpu), label.to(args.gpu)
        label -= 1
        # 生成扩展域和中间域 no_grad
        with torch.no_grad():
            x_ED = generator(data)
        rand = torch.nn.init.uniform_(torch.empty(len(data), 1, 1, 1)).to(args.gpu)
        x_ID = rand * data + (1 - rand) * x_ED

        x_TD = generator(data)  # .detach()
        label_TD = label

        #         data = torch.cat((data, x_ED), dim=0)
        #         label = label.repeat(2)
        out_SD, band_SD, feat_SD = model(data, mode='train')
        out_ED, band_ED, feat_ED = model(x_ED, mode='train')

        assert feat_SD.size(0) == feat_ED.size(0)
        out_TD = model(x_TD)

        cls_loss_TD = cls_criterion(out_TD, label_TD)  # TD分类损失

        cls_loss = cls_criterion(out_SD, label) + cls_criterion(out_ED, label)  # 分类损失

        loss_fac = factorization_loss(feat_SD, feat_ED)
        loss_cc = cc_criterion(band_SD, label) + cc_criterion(band_ED, label)
        loss = 0.8 * cls_loss + 0.2 * cls_loss_TD + loss_cc + loss_fac    # loss_cc刚开始会非常大
        # 指标前期不变原因未查明
        M_opt.zero_grad()
        C_opt.zero_grad()
        loss.backward(retain_graph=True)

        G_opt.zero_grad()
        loss = 0.2 * cls_loss + 0.8 * cls_loss_TD
        loss.backward()
        M_opt.step()
        C_opt.step()
        G_opt.step()

        # 打印统计信息
        train_loss += cls_loss.item()

        writer.add_scalar('cls_loss', cls_loss, epoch)

        if i % 10 == 9:  # 每10个小批次打印一次
            # 更新训练信息
            loop.set_description('Epoch [{}/{}]'.format(epoch + 1, args.epoch))
            loop.set_postfix(loss=train_loss / 10)
            train_loss = 0.0


def validation(best_acc):
    model.eval()
    ps = []
    ys = []
    loop = tqdm(test_loader, desc='Testing')
    for i, (x1, y1) in enumerate(loop):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(args.gpu)
            p1 = model(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    results = metrics(ps, ys, n_classes=ys.max() + 1)
    print('TPR: {} | current OA: {:2.2f} | best OA: {:2.2f}'.format(np.round(results['TPR'] * 100, 2),
                                                                    results['Accuracy'], best_acc))
    return acc


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
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_domain,
                                                                                        os.path.join(DATA_ROOT,
                                                                                                     args.data_path))
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_domain,
                                                                                        os.path.join(DATA_ROOT,
                                                                                                     args.data_path))

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

    # 创建模型和优化器
    # 判别器&优化器
    model = CausalNet(in_channels=N_BANDS, out_channels=hyperparams['pro_dim'], num_classes=num_classes)
    M_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 生成器&优化器
    generator = Generator(n=args.d_se, imdim=N_BANDS, imsize=imsize, zdim=10, device=args.gpu)
    G_opt = Adam(generator.parameters(), lr=args.lr)

    # loss
    cc_criterion = CategoryConsistencyLoss(num_classes=num_classes, embedding_size=N_BANDS, device=args.gpu)
    C_opt = torch.optim.Adam(cc_criterion.parameters(), lr=args.lr)

    cls_criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.to(args.gpu)
        generator.to(args.gpu)
        cc_criterion.to(args.gpu)

    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        print('-' * 45 + 'Training' + '-' * 45)
        start = time.time()
        train(epoch)
        end = time.time()
        print('epoch time:', end - start)
        print('-' * 44 + 'Validating' + '-' * 44)
        taracc = validation(best_acc=best_acc)
        if best_acc < taracc:
            best_acc = taracc
            torch.save({'Discriminator': model.state_dict()}, os.path.join(log_dir, f'best.pkl'))
    writer.close()
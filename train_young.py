#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CausalExtension 
@File    ：train_young.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2024/10/13 10:47 
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
from models.con_loss import SupConLoss
from models.discriminator import Discriminator
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


def train(best_acc):
    model.train()
    loss_list = []
    train_loss = 0  # loss
    loop = tqdm(train_loader, desc='Epoch [{}/{}]'.format(epoch + 1, args.epoch))
    for i, (data, label) in enumerate(loop):
        data, label = data.cuda(), label.cuda()
        label -= 1
        # 生成扩展域和中间域 no_grad
        with torch.no_grad():
            x_ED = generator(data)
        rand = torch.nn.init.uniform_(torch.empty(len(data), 1, 1, 1)).to(args.gpu)
        x_ID = rand * data + (1 - rand) * x_ED

        x_tgt = generator(data)#.detach()
        ed_label = label

        data = torch.cat((data, x_ED), dim=0)
        label = label.repeat(2)
        out, band_weights, features = model(data)
        split_idx = int(data.size(0) / 2)
        features_ori, features_ed = torch.split(features, split_idx)
        assert features_ori.size(0) == features_ed.size(0)

        ed_out, _, _ = model(x_tgt)

        ed_cls_loss = cls_criterion(ed_out, ed_label)  # ed分类损失
        cls_loss = cls_criterion(out, label)  # 分类损失
        loss_cc = cc_criterion(band_weights, label)
        loss_fac = factorization_loss(features_ori, features_ed)
        # loss = 0.8 * cls_loss + 0.2 * ed_cls_loss + 0.01 * loss_cc
        loss = 0.8 * cls_loss + 0.2 * ed_cls_loss + 0.01 * loss_cc + 0.5 * loss_fac
        # optimizers['extractor'].zero_grad()
        # optimizers['classifier'].zero_grad()
        M_opt.zero_grad()

        # 反向传播
        loss.backward(retain_graph=True)

        G_opt.zero_grad()

        # loss = 0.8 * cls_loss + 0.2 * ed_cls_loss + 0.5 * loss_fac
        loss = 0.2 * cls_loss + 0.8 * ed_cls_loss

        loss.backward()
        M_opt.step()
        G_opt.step()

        # 打印统计信息
        train_loss += cls_loss.item()

        writer.add_scalar('cls_loss', cls_loss, epoch)

        if i % 10 == 9:  # 每10个小批次打印一次
            # 更新训练信息
            loop.set_description('Epoch [{}/{}]'.format(epoch + 1, args.epoch))
            loop.set_postfix(loss=train_loss / 10)
            train_loss = 0.0


def evaluate(best_acc):

    # extractor = Extractor(102, 256)
    # classifier = Classifier(input_size=256, num_classes=7)
    # model = CausalNet(in_channels=102, num_classes=7)
    #
    # if torch.cuda.is_available():
    #     state_dict = torch.load("./run/Pavia/ckpt/best.pth.tar")
    #     # extractor.cuda(), classifier.cuda()
    #     model.cuda()
    # else:
    #     state_dict = torch.load("./run/Pavia/ckpt/best.pth.tar", map_location=torch.device('cpu'))

    # extractor.load_state_dict(state_dict['extractor_state_dict'])
    # classifier.load_state_dict(state_dict['classifier_state_dict'])
    # model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    loop = tqdm(test_loader, desc='Testing')
    with torch.no_grad():
        outs = []
        labels = []
        for i, (data, label) in enumerate(loop):
            label = label - 1
            # GPU processing
            data = data.cuda()
            # features = extractor(data)
            # out = classifier(features)
            out, _, features = model(data)
            out = out.argmax(dim=1)
            outs.append(out.detach().cpu().numpy())
            labels.append(label.numpy())

        outs = np.concatenate(outs)
        labels = np.concatenate(labels)
        acc = np.mean(outs == labels) * 100

        metrics = get_metrics(outs, labels)

        print('metrics [oa: {:2.4f} | kappa: {:2.4f} | aa: {:2.4f} | test_acc: {:2.2f} | best_acc: {:2.2f}]'.format(metrics['oa'],
                                                                                                metrics['kappa'],
                                                                                                metrics['aa'], acc, best_acc))
        if acc > best_acc:
            best_acc = acc
            save_model(model=model, best_acc=best_acc, epoch=epoch, folder=args.source_domain, best=True)
            print('[Saving Best Snapshot:] run/{}/ckpt/best.pth.tar'.format(args.source_domain))
        return best_acc

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

    num_bands = img_src.shape[0]  # 通道数
    num_classes = gt_src.max()  # 类别数

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



    start_epoch = 0
    best_acc = 0.0
    model = CausalNet(in_channels=num_bands, num_classes=num_classes)
    M_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    cc_criterion = CategoryConsistencyLoss(num_classes=num_classes, embedding_size=num_bands)
    C_opt = torch.optim.Adam(cc_criterion.parameters(), lr=args.lr)
    # 生成器&优化器
    generator = Generator(n=args.d_se, imdim=N_BANDS, imsize=imsize, zdim=10, device=args.gpu)
    G_opt = Adam(generator.parameters(), lr=args.lr)

    # loss
    cls_criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        generator.cuda()
        cc_criterion.cuda()
    for epoch in range(start_epoch, args.epoch):
        print('-' * 45 + 'Training' + '-' * 45)
        t1 = time.time()
        train(epoch)
        t2 = time.time()
        print('epoch time:', t2 - t1)
        print('-' * 44 + 'Validating' + '-' * 44)
        # best_acc = validate(epoch, models=models, val_loader=val_loader, best_acc=best_acc)
        best_acc = evaluate(best_acc=best_acc)
    writer.close()
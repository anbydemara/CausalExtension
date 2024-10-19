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


def train(best_acc):
    D_net.train()
    loss_list = []
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(args.gpu), y.to(args.gpu)
        y = y - 1
        # 生成扩展域和中间域 no_grad
        with torch.no_grad():
            x_ED = G_net(x)
        rand = torch.nn.init.uniform_(torch.empty(len(x), 1, 1, 1)).to(args.gpu)
        x_ID = rand * x + (1 - rand) * x_ED

        x_tgt = G_net(x)#.detach()

        p_SD, z_SD = D_net(x, mode='train')  # p_:分类头， z_投影头
        p_ED, z_ED = D_net(x_ED, mode='train')
        p_ID, z_ID = D_net(x_ID, mode='train')

        zsrc = torch.cat([z_SD.unsqueeze(1), z_ED.unsqueeze(1), z_ID.unsqueeze(1)], dim=1)
        src_cls_loss = cls_criterion(p_SD, y.long()) + cls_criterion(p_ED, y.long()) + cls_criterion(p_ID, y.long())

        ## ---------------------
        ##  Train Discriminator
        ## ---------------------
        p_tgt, z_tgt = D_net(x_tgt, mode='train')
        tgt_cls_loss = cls_criterion(p_tgt, y.long())

        zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1)
        con_loss = con_criterion(zall, y, adv=False)
        loss = src_cls_loss + args.lambda_1 * con_loss + tgt_cls_loss

        D_opt.zero_grad()
        loss.backward(retain_graph=True)

        num_adv = y.unique().size()
        # zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(1), z_ID.unsqueeze(1)], dim=1)
        zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(1).detach(), z_ID.unsqueeze(1).detach()], dim=1)
        con_loss_adv = 0
        idx_1 = np.random.randint(0, zsrc.size(1))

        ## ---------------------
        ##  Train Generator
        ## ---------------------
        for i, id in enumerate(y.unique()):
            mask = y == y.unique()[i]
            z_SD_i, zsrc_i = z_SD[mask], zsrc_con[mask]
            y_i = torch.cat([torch.zeros(z_SD_i.shape[0]), torch.ones(z_SD_i.shape[0])])
            zall = torch.cat([z_SD_i.unsqueeze(1).detach(), zsrc_i[:, idx_1:idx_1 + 1]], dim=0)
            if y_i.size()[0] > 2:
                con_loss_adv += con_criterion(zall, y_i)
        con_loss_adv = con_loss_adv / y.unique().shape[0]

        loss = tgt_cls_loss + args.lambda_2 * con_loss_adv
        G_opt.zero_grad()
        loss.backward()
        D_opt.step()
        G_opt.step()
        # if args.lr_scheduler in ['cosine']:
        #     scheduler.step()
        loss_list.append([src_cls_loss.item(), tgt_cls_loss.item(), con_loss.item(), con_loss_adv.item()])

    src_cls_loss, tgt_cls_loss, con_loss, con_loss_adv = np.mean(loss_list, 0)
    D_net.eval()
    teacc = evaluate(D_net, val_loader, args.gpu)
    if best_acc < teacc:
        best_acc = teacc
        torch.save({'Discriminator': D_net.state_dict()}, os.path.join(log_dir, f'best.pkl'))

    print(
        f'epoch {epoch}, train {len(train_loader.dataset)}, src_cls {src_cls_loss:.4f} tgt_cls {tgt_cls_loss:.4f} con {con_loss:.4f} con_adv {con_loss_adv:.4f} /// val {len(val_loader.dataset)}, teacc {teacc:2.2f}')
    writer.add_scalar('src_cls_loss', src_cls_loss, epoch)
    writer.add_scalar('tgt_cls_loss', tgt_cls_loss, epoch)
    writer.add_scalar('con_loss', con_loss, epoch)
    writer.add_scalar('con_loss_adv', con_loss_adv, epoch)
    writer.add_scalar('teacc', teacc, epoch)

    if epoch % args.log_interval == 0:
        pklpath = f'{log_dir}/best.pkl'
        taracc = evaluate_tgt(D_net, args.gpu, test_loader, pklpath)
        taracc_list.append(round(taracc, 2))
        print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')


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

    # 判别器&优化器
    D_net = Discriminator(inchannel=N_BANDS, outchannel=args.pro_dim, num_classes=num_classes,
                          patch_size=hyperparams['patch_size']).to(args.gpu)
    D_opt = Adam(D_net.parameters(), lr=args.lr)

    # 生成器&优化器
    G_net = Generator(n=args.d_se, imdim=N_BANDS, imsize=imsize, zdim=10, device=args.gpu).to(args.gpu)
    G_opt = Adam(G_net.parameters(), lr=args.lr)

    # loss
    cls_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(device=args.gpu)

    best_acc = 0
    taracc, taracc_list = 0, []
    for epoch in range(1, args.epoch + 1):
        start = time.time()
        train(best_acc=best_acc)
        end = time.time()
    writer.close()
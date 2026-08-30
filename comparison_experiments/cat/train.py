"""CAT (Cluster Alignment with a Teacher) baseline。

Deng, Luo, Zhu, "Cluster Alignment with a Teacher for Unsupervised Domain
Adaptation", ICCV 2019 (arXiv:1903.09980)。DAGCN 论文（TMI 2025）对比方法，
ref [41]（"cluster and teacher alignment"）。

★ 按 OCT-DDA 官方代码（xuqing88/OCT_DDA，`CAT_vgg16_train.py`）完整对齐
  （与 EM-DDA/ADDA 同源同配置）：
  - 编码器：vgg16_bn.features（卷积特征，展平 25088 维）
  - 分类器：vgg16_bn 原生 classifier（fc6-fc7-fc8'，fc8'=n_cls）
  - 判别器 netD：Linear(3→200)+ReLU + Linear(200→200)+ReLU + Linear(200,1)+Sigmoid
    （输入 = 3 维 logits）
  - loss = loss_cls + lamb2·sntg_loss（官方；对应论文 Eq.4 的 L_y + α(L_c+L_a)）
        sntg_loss = L_a(类原型对齐) + L_c(目标) + L_c(源)，全部在 logits 上
        L_c(源)：真实标签同类指示 δ，平方欧氏距离，margin LAMBDA=0.01（官方取值，
                 注释注明论文为 30）
        L_c(目标)：teacher 伪标签（Π-model：目标样本两次前向，第二次 detach）
        L_a：源/目标类原型（logits 均值）L2 距离，fm_mask 剔除缺失类并归一化
        lamb2 ramp-up：gd≥5 起，lamb2 = exp(-(1-min((gd-5)/15,1))·10)（第 20 epoch≈1）
  - RevGrad 式域对抗（0.1 权重）：D 用 MultiLabelSoftMarginLoss 判别源/目标 logits，
    D_loss=0.1·D_loss，与主 loss 分步 backward（loss.backward(retain_graph=True);
    D_loss.backward()；★ 严格对齐官方——官方无 G_loss=-D_loss，旧实现符号相反会使
    判别器往"判别越错"方向训练，已修复 2026-08-09）
  - SGD lr=0.001（encoder+classifier 0.001 / netD 固定 0.001, momentum=0.9）；batch=8；epochs=15
    （★ 2026-08-09：官方 30 → 15 轮早停，lamb2 ramp-up 至 1.0 后崩溃，epoch15≈0.89 为峰）
  - transform：Resize(256)+CenterCrop(224)+ToTensor（官方无 Normalize）

Usage:
    python -m comparison_experiments.cat.train --src A --tgt B --seed 777
"""
import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.common.data_loader import load_task, set_seed, LABEL_TO_DATASET
from comparison_experiments.common.evaluate import test
from comparison_experiments.common.models import VGGBnBackbone, VGGBnClassifier


def adaptation_factor(x):
    """官方 adaptation_factor：2/(1+exp(-10x))-1，截断 ≤1。"""
    lamb = 2.0 / (1.0 + math.exp(-10.0 * x)) - 1.0
    return min(lamb, 1.0)


def unsorted_segment_sum(data, segment_ids, num_segments, device):
    """按 segment_ids 求和（官方 util_cat.py `unsorted_segment_sum_device` 等价实现）。

    data 可为 1D（计数）或 2D [N, D]（logits/特征）；segment_ids 1D [N]。
    返回 [num_segments] 或 [num_segments, D]。
    """
    if len(segment_ids.shape) == 1 and data.dim() > 1:
        seg = segment_ids.unsqueeze(1).expand_as(data)
    else:
        seg = segment_ids
    shape = [num_segments] + list(data.shape[1:])
    out = torch.zeros(*shape, device=device)
    out.scatter_add_(0, seg, data.float())
    return out.type(data.dtype)


class Discriminator(nn.Module):
    """OCT-DDA 官方 CAT 判别器：输入 3 维 logits，Linear(3→200)+ReLU ×2 + Linear(200,1)+Sigmoid。"""

    def __init__(self, input_features=3, hidden_size=200):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x


def train(args):
    set_seed(args.seed)
    # 官方 transform 无 ImageNet Normalize
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch, normalize=False)
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))
    print("  [CAT] OCT-DDA 官方：vgg16_bn.features(25088) + logits 上 SNTG/原型 + Π-model teacher")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = VGGBnBackbone().to(device)
    clf = VGGBnClassifier(num_classes=n_cls).to(device)
    netD = Discriminator(input_features=n_cls).to(device)
    print("  [CAT] models ready (feature dim=25088, discriminator on logits)")

    # 官方超参：lr=0.001；LAMBDA=0.01（margin，注释注明原论文为 30）
    LAMBDA = args.margin            # 0.01（官方）
    lamb2_rampup = 5                # sntg 权重 ramp-up 起始 epoch
    lamb2_rampup_win = 15           # ramp-up 窗口（第 5→20 epoch 升到 ~1）

    optimizer = optim.SGD(list(enc.parameters()) + list(clf.parameters()),
                          lr=args.lr, momentum=0.9)
    # ★ 官方对齐：判别器 lr 固定 0.001（官方 CAT_vgg16_train.py 独立于主 lr）
    optimizerD = optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    src_train = data['src_train']
    tgt_train = data['tgt_train']
    n_iter = min(len(src_train), len(tgt_train))

    gd = 0
    for epoch in range(args.epochs):
        enc.train(); clf.train(); netD.train()
        gd += 1
        # sntg 权重：第 5 epoch 起 ramp-up，第 20 epoch ≈ 1.0
        lamb2 = math.exp(-(1.0 - min((gd - lamb2_rampup) * 1.0 / lamb2_rampup_win, 1.0)) * 10.0) \
            if gd >= lamb2_rampup else 0.0
        lamb = adaptation_factor(gd * 1.0 / args.epochs)   # 未使用（官方保留变量）
        iter_src = iter(src_train)
        iter_tgt = iter(tgt_train)

        for i in range(n_iter):
            xs, ys = next(iter_src)
            xt, _ = next(iter_tgt)
            if xs.size(0) != xt.size(0):
                continue   # 等价官方 drop_last：跳过源/目标 batch 大小不等的一批（onehot/scatter 需同形）
            if torch.cuda.is_available():
                xs, ys, xt = xs.cuda(), ys.cuda(), xt.cuda()
            bs = xs.size(0)

            optimizer.zero_grad()
            optimizerD.zero_grad()

            # ── 分类损失（源域）──
            source_logits = clf(enc(xs))                     # [B, n_cls] 即论文 "feature space"
            loss_cls = criterion(source_logits, ys)

            # ── 目标：Π-model teacher（两次前向，第二次 detach 作伪标签）──
            target_logits = clf(enc(xt))                     # 带梯度
            with torch.no_grad():
                target_logits_2 = clf(enc(xt)).detach()      # teacher（dropout 扰动）
            target_pred = target_logits_2.argmax(dim=1)
            target_onehot = torch.zeros(bs, n_cls, device=device).scatter_(1, target_pred.unsqueeze(-1), 1)

            # ── 域判别（输入 = logits）──
            D_src = netD(source_logits)                      # [B, 1]
            D_tgt = netD(target_logits)
            D_real_loss = torch.mean(nn.MultiLabelSoftMarginLoss()(
                D_tgt, torch.ones_like(D_tgt).to(device)))
            D_fake_loss = torch.mean(nn.MultiLabelSoftMarginLoss()(
                D_src, torch.zeros_like(D_src).to(device)))
            # ★ 官方对齐：D_loss 用 0.1 缩放（RevGrad 惯例），backward 走 D_loss 本身
            #   （官方 CAT_vgg16_train.py 只有 loss.backward(retain_graph) + D_loss.backward()，
            #    无 G_loss=-D_loss。原实现符号相反会使判别器往"判别越错"方向训练）
            D_loss = 0.1 * (D_real_loss + D_fake_loss)

            # ── L_c(源)：真实标签同类指示 δ ──
            y_onehot = torch.zeros(bs, n_cls, device=device).scatter_(
                1, ys.unsqueeze(-1), 1)
            graph_src = torch.sum(y_onehot[:, None, :] * y_onehot[None, :, :], dim=2)
            dist_src = torch.mean((source_logits[:, None, :] - source_logits[None, :, :]) ** 2, dim=2)
            source_sntg = torch.mean(graph_src * dist_src
                                     + (1 - graph_src) * torch.relu(LAMBDA - dist_src))

            # ── L_c(目标)：teacher 伪标签同类指示 ──
            graph_tgt = torch.sum(target_onehot[:, None, :] * target_onehot[None, :, :], dim=2)
            dist_tgt = torch.mean((target_logits[:, None, :] - target_logits[None, :, :]) ** 2, dim=2)
            target_sntg = torch.mean(graph_tgt * dist_tgt
                                     + (1 - graph_tgt) * torch.relu(LAMBDA - dist_tgt))

            # ── L_a：类原型对齐（fm_mask 剔除缺失类并归一化）──
            src_label = ys
            cur_src_count = unsorted_segment_sum(torch.ones_like(src_label, dtype=torch.float32, device=device),
                                                 src_label, n_cls, device)
            cur_tgt_count = unsorted_segment_sum(torch.ones_like(target_pred, dtype=torch.float32, device=device),
                                                 target_pred, n_cls, device)
            cur_src_centroid = unsorted_segment_sum(source_logits, src_label, n_cls, device) / \
                torch.max(cur_src_count, torch.ones_like(cur_src_count).to(device))[:, None]
            cur_tgt_centroid = unsorted_segment_sum(target_logits, target_pred, n_cls, device) / \
                torch.max(cur_tgt_count, torch.ones_like(cur_tgt_count).to(device))[:, None]
            fm_mask = torch.gt(cur_src_count * cur_tgt_count, 0).float()
            fm_mask = fm_mask / (torch.mean(fm_mask + 1e-8))
            L_a = torch.mean(torch.mean((cur_src_centroid - cur_tgt_centroid) ** 2, 1) * fm_mask)

            sntg_loss = L_a + target_sntg + source_sntg
            loss = loss_cls + lamb2 * sntg_loss

            # ★ 官方对齐：loss 主损失 + D_loss（含 0.1 缩放）分别 backward
            #   （官方 CAT_vgg16_train.py：loss.backward(retain_graph=True); D_loss.backward()）
            loss.backward(retain_graph=True)
            D_loss.backward()
            optimizer.step()
            optimizerD.step()

        acc, auc, _ = test(enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                           num_classes=n_cls, class_names=data['class_names'])
        print("  [CAT] epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f} (lamb2={:.3f})".format(
            epoch + 1, args.epochs, acc, auc, lamb2))


def main():
    parser = argparse.ArgumentParser(description='CAT baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=15,
                        help='OCT-DDA 官方默认 30；本项目 15 轮早停（lamb2 ramp-up 至 1.0 后崩溃，用户 2026-08-09 决定）')
    parser.add_argument('--batch', type=int, default=8,
                        help='OCT-DDA 官方 batch=8')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='OCT-DDA 官方 CAT lr=0.001')
    parser.add_argument('--margin', type=float, default=0.01,
                        help='OCT-DDA 官方 LAMBDA=0.01（注释注明论文原值 30）')
    parser.add_argument('--input_size', type=int, default=224)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

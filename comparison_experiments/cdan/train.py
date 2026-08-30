"""CDAN (Conditional Adversarial Domain Adaptation) baseline.

Long et al., "Conditional Adversarial Domain Adaptation", NeurIPS 2018.
按 thuml/Transfer-Learning-Library 官方复现（alignment/cdan.py + example/cdan.py）：
  - MultiLinearMap：f ⊗ g（bottleneck 256 特征 × C 类）展平
  - 判别器：thuml DomainDiscriminator（sigmoid 单输出 + BN + 1024 hidden）
  - WarmStart GRL + 单一 SGD optimizer（分层 lr，backbone 0.1×）
  - CDAN-E：entropy conditioning（--entropy 时启用）
  - 超参：lr=0.01, lr_gamma=0.001, lr_decay=0.75, wd=1e-3, trade_off=1.0

Usage:
    python -m comparison_experiments.cdan.train --src A --tgt B --seed 777
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.common.data_loader import load_task, set_seed, LABEL_TO_DATASET
from comparison_experiments.common.evaluate import test
from comparison_experiments.common.models import build_models


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class WarmStartGradientReverseLayer(nn.Module):
    """thuml WarmStartGradientReverseLayer（alpha 0→1，max_iters=1000）。"""

    def __init__(self, alpha=1.0, lo=0.0, hi=1.0, max_iters=1000., auto_step=True):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, x):
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(x, self.alpha)

    def step(self):
        self.iter_num += 1
        if self.iter_num < self.max_iters:
            self.alpha = 2.0 * (self.hi - self.lo) / (1.0 + np.exp(-10.0 * self.iter_num / self.max_iters)) \
                         - (self.hi - self.lo) + self.lo
        else:
            self.alpha = self.hi


class DomainDiscriminator(nn.Module):
    """thuml 官方 DomainDiscriminator（sigmoid 单输出 + BN + hidden 1024）。"""

    def __init__(self, in_feature, hidden_size=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feature, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

    def get_parameters(self, base_lr=1.0):
        return [{"params": self.parameters(), "lr": 1.0 * base_lr}]


class MultiLinearMap(nn.Module):
    """thuml MultiLinearMap：T(f, g) = f ⊗ g 展平（官方 cdan.py）。"""

    def forward(self, f, g):
        return torch.bmm(f.unsqueeze(2), g.unsqueeze(1)).view(f.size(0), -1)


def entropy(p):
    """对 softmax 概率计算熵（thuml modules/entropy.py）。"""
    return -(p * torch.log(p + 1e-5)).sum(dim=1)


class ConditionalDomainAdversarialLoss(nn.Module):
    """thuml 官方 CDAN 损失（alignment/cdan.py），sigmoid + BCE。"""

    def __init__(self, domain_discriminator, entropy_conditioning=False, reduction='mean'):
        super().__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.map = MultiLinearMap()
        self.entropy_conditioning = entropy_conditioning
        self.reduction = reduction

    def forward(self, g_s, f_s, g_t, f_t):
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)
        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size
        d_label = torch.cat((torch.ones((g_s.size(0), 1)).to(g_s.device),
                             torch.zeros((g_t.size(0), 1)).to(g_t.device)))
        if self.entropy_conditioning:
            return F.binary_cross_entropy(d, d_label, weight.view_as(d), reduction=self.reduction)
        else:
            return F.binary_cross_entropy(d, d_label, reduction=self.reduction)


def train(args):
    set_seed(args.seed)
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch)
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc, clf = build_models(n_cls, device)   # bottleneck 'bn' 官方
    # CDAN 官方：判别器输入 = features_dim * num_classes（bottleneck 256 × C）
    disc = DomainDiscriminator(in_feature=clf.features_dim * n_cls).to(device)

    # thuml 官方：单一 SGD，分层 lr（backbone 0.1×），nesterov + wd
    optimizer = optim.SGD(clf.get_parameters(base_lr=1.0, backbone=enc) + disc.get_parameters(),
                          lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer,
                            lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    cdan_loss = ConditionalDomainAdversarialLoss(disc, entropy_conditioning=args.entropy).to(device)
    ce = nn.CrossEntropyLoss()

    src_train = data['src_train']
    tgt_train = data['tgt_train']
    n_iter = min(len(src_train), len(tgt_train))

    for epoch in range(args.epochs):
        enc.train(); clf.train(); disc.train()
        it_s = iter(src_train)
        it_t = iter(tgt_train)
        for i in range(n_iter):
            xs, ys = next(it_s)
            xt, _ = next(it_t)
            if torch.cuda.is_available():
                xs, ys, xt = xs.cuda(), ys.cuda(), xt.cuda()

            optimizer.zero_grad()
            fs = enc(xs)
            ft = enc(xt)
            logits_s = clf(fs)
            logits_t = clf(ft)
            loss_cls = ce(logits_s, ys)
            f256_s = clf.forward_features(fs)
            f256_t = clf.forward_features(ft)
            loss_adv = cdan_loss(logits_s, f256_s, logits_t, f256_t)
            loss = loss_cls + args.trade_off * loss_adv
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        acc, auc, _ = test(enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                           num_classes=n_cls, class_names=data['class_names'])
        print("  [CDAN] epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f}".format(epoch + 1, args.epochs, acc, auc))


def main():
    parser = argparse.ArgumentParser(description='CDAN baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=20,
                        help='thuml 官方 CDAN epochs=20')
    parser.add_argument('--batch', type=int, default=32,
                        help='thuml 官方 CDAN batch=32')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='thuml 官方 CDAN lr=0.01')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='thuml 官方 CDAN wd=1e-3')
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--trade_off', type=float, default=1.0)
    parser.add_argument('--entropy', action='store_true',
                        help='使用 CDAN-E 熵条件加权（thuml 官方 --entropy）')
    parser.add_argument('--input_size', type=int, default=224,
                        help='thuml ResNet 官方输入 224')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

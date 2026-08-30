"""DANN (Domain-Adversarial Neural Network) baseline.

Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
Code adapted from thuml/Transfer-Learning-Library for OCT cross-device UDA.

Adversarial representative baseline: gradient reversal layer + domain discriminator.
Usage:
    python -m comparison_experiments.dann.train --src A --tgt B --seed 777
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
    """thuml GradientReverseFunction：forward 恒等，backward 乘 -alpha。"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class WarmStartGradientReverseLayer(nn.Module):
    """thuml WarmStartGradientReverseLayer（modules/grl.py）。

    alpha 从 lo 渐变到 hi（sigmoid 型，max_iters 内），自动按累计迭代步推进。
    DANN 官方 (thuml) 默认 alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True。
    """

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
        """渐增 alpha：2(hi-lo)/(1+exp(-10*iter/max)) - (hi-lo) + lo，iter<max 前。"""
        self.iter_num += 1
        if self.iter_num < self.max_iters:
            self.alpha = 2.0 * (self.hi - self.lo) / (1.0 + np.exp(-10.0 * self.iter_num / self.max_iters)) \
                         - (self.hi - self.lo) + self.lo
        else:
            self.alpha = self.hi


class DomainDiscriminator(nn.Module):
    """thuml 官方 DomainDiscriminator（modules/domain_discriminator.py）。

    Linear(256->1024)+BN+ReLU + Linear(1024->1024)+BN+ReLU + Linear(1024,1)+Sigmoid。
    输入 = bottleneck 后 256 维特征；输出 1 维 sigmoid（源=1/目标=0）。
    """

    def __init__(self, in_feature=256, hidden_size=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feature, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

    def get_parameters(self, base_lr=1.0):
        return [{"params": self.parameters(), "lr": 1.0 * base_lr}]


class DomainAdversarialLoss(nn.Module):
    """thuml 官方 DomainAdversarialLoss（alignment/dann.py）。

    判别器+目标编码器在同一个 optimizer 内更新：GRL 自动反转特征梯度，
    BCE 区分源(1)/目标(0)。reduction='mean'。
    """

    def __init__(self, domain_discriminator, reduction='mean', grl=None):
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                                                 max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.reduction = reduction

    def forward(self, f_s, f_t):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        # 官方 chunk(2) 假设源/目标 batch 相同；A-B 源/目标最后一批可能不等（如 14 vs 12），
        # 会切错导致 label 与预测尺寸不匹配而崩溃 → 按实际 batch 大小切片
        bs = f_s.size(0)
        d_s, d_t = d[:bs], d[bs:]
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        return 0.5 * (F.binary_cross_entropy(d_s, d_label_s, reduction=self.reduction) +
                      F.binary_cross_entropy(d_t, d_label_t, reduction=self.reduction))


def train(args):
    set_seed(args.seed)
    print("[DANN] loading data ...", flush=True)
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch)
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[DANN] device={}  loading ResNet50 weights ...".format(device), flush=True)
    enc, clf = build_models(len(data['class_names']), device)
    # thuml 官方：判别器输入 = bottleneck 后 256 维特征
    disc = DomainDiscriminator(in_feature=clf.features_dim).to(device)
    print("[DANN] models ready (bottleneck dim={})".format(clf.features_dim), flush=True)

    # thuml 官方优化器：单一 SGD，backbone 0.1×lr，bottleneck/head 1×lr，
    # nesterov + weight_decay；判别器同一 optimizer（GRL 反转梯度）
    param_groups = clf.get_parameters(base_lr=1.0, backbone=enc) + disc.get_parameters()
    optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    # thuml 官方 lr_scheduler: lr * (1 + lr_gamma * step) ** (-lr_decay)
    lr_scheduler = LambdaLR(optimizer,
                            lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    domain_adv = DomainAdversarialLoss(disc).to(device)
    ce = nn.CrossEntropyLoss()

    src_train = data['src_train']
    tgt_train = data['tgt_train']
    n_iter = min(len(src_train), len(tgt_train))
    global_step = 0

    for epoch in range(args.epochs):
        enc.train(); clf.train(); disc.train()
        iter_src = iter(src_train)
        iter_tgt = iter(tgt_train)
        for i in range(n_iter):
            xs, ys = next(iter_src)
            xt, _ = next(iter_tgt)
            if torch.cuda.is_available():
                xs, ys, xt = xs.cuda(), ys.cuda(), xt.cuda()

            optimizer.zero_grad()
            fs = enc(xs)
            ft = enc(xt)
            logits_s = clf(fs)
            loss_cls = ce(logits_s, ys)
            # thuml 官方：判别器作用在 bottleneck 后 256 维特征
            f256_s = clf.forward_features(fs)
            f256_t = clf.forward_features(ft)
            loss_adv = domain_adv(f256_s, f256_t)
            loss = loss_cls + loss_adv
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

        acc, auc, _ = test(enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                           num_classes=len(data['class_names']), class_names=data['class_names'])
        print("  [DANN] epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f}".format(epoch + 1, args.epochs, acc, auc))


def main():
    parser = argparse.ArgumentParser(description='DANN baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='thuml 官方 DANN lr=0.01（TLL examples/.../dann.py）')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='thuml 官方 DANN weight_decay=1e-3')
    parser.add_argument('--lr_gamma', type=float, default=0.001,
                        help='thuml 官方 lr_scheduler gamma（LambdaLR: (1+γx)^(-decay)）')
    parser.add_argument('--lr_decay', type=float, default=0.75,
                        help='thuml 官方 lr_scheduler decay')
    parser.add_argument('--input_size', type=int, default=224,
                        help='thuml ResNet 官方输入 224')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

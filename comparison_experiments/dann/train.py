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
    """thuml GradientReverseFunction: identity forward, backward scaled by -alpha."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class WarmStartGradientReverseLayer(nn.Module):
    """thuml WarmStartGradientReverseLayer (modules/grl.py).

    alpha ramps from lo to hi (sigmoid-shaped within max_iters), auto-stepping per iteration.
    DANN official (thuml) defaults: alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True.
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
        """Ramp alpha: 2(hi-lo)/(1+exp(-10*iter/max)) - (hi-lo) + lo while iter < max."""
        self.iter_num += 1
        if self.iter_num < self.max_iters:
            self.alpha = 2.0 * (self.hi - self.lo) / (1.0 + np.exp(-10.0 * self.iter_num / self.max_iters)) \
                         - (self.hi - self.lo) + self.lo
        else:
            self.alpha = self.hi


class DomainDiscriminator(nn.Module):
    """thuml official DomainDiscriminator (modules/domain_discriminator.py).

    Linear(256->1024)+BN+ReLU + Linear(1024->1024)+BN+ReLU + Linear(1024,1)+Sigmoid.
    Input = 256-d bottleneck features; output = 1-d sigmoid (source=1 / target=0).
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
    """thuml official DomainAdversarialLoss (alignment/dann.py).

    Discriminator and encoder share one optimizer: GRL reverses feature gradients,
    BCE separates source (1) / target (0). reduction='mean'.
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
        # official chunk(2) assumes equal src/tgt batch; slice by actual sizes instead
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
    # official: discriminator input = 256-d bottleneck features
    disc = DomainDiscriminator(in_feature=clf.features_dim).to(device)
    print("[DANN] models ready (bottleneck dim={})".format(clf.features_dim), flush=True)

    # official optimizer: single SGD, backbone 0.1x lr, bottleneck/head 1x lr,
    # nesterov + weight_decay; discriminator in the same optimizer (GRL reverses grad)
    param_groups = clf.get_parameters(base_lr=1.0, backbone=enc) + disc.get_parameters()
    optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    # official lr_scheduler: lr * (1 + lr_gamma * step) ** (-lr_decay)
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
            # official: discriminator operates on 256-d bottleneck features
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
                        help='thuml official DANN lr=0.01 (TLL examples/.../dann.py)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='thuml official DANN weight_decay=1e-3')
    parser.add_argument('--lr_gamma', type=float, default=0.001,
                        help='thuml official lr_scheduler gamma (LambdaLR: (1+γx)^(-decay))')
    parser.add_argument('--lr_decay', type=float, default=0.75,
                        help='thuml official lr_scheduler decay')
    parser.add_argument('--input_size', type=int, default=224,
                        help='thuml ResNet official input 224')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

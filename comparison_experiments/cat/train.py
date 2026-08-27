"""CAT (Cluster Alignment with a Teacher) baseline.

Deng, Luo, Zhu, "Cluster Alignment with a Teacher for Unsupervised Domain
Adaptation", ICCV 2019. Aligned with the OCT-DDA official code (CAT_vgg16_train.py):
encoder vgg16_bn.features (25088-d); discriminator on 3-d logits (3->200->200->1+sigmoid);
loss = loss_cls + lamb2*sntg_loss; RevGrad-style adversarial (0.1 weight); SGD lr=0.001, batch 8.

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
    """Official adaptation_factor: 2/(1+exp(-10x))-1, clipped to <=1."""
    lamb = 2.0 / (1.0 + math.exp(-10.0 * x)) - 1.0
    return min(lamb, 1.0)


def unsorted_segment_sum(data, segment_ids, num_segments, device):
    """Sum by segment_ids (equivalent of util_cat.py unsorted_segment_sum_device).

    data: 1D (counts) or 2D [N, D]; segment_ids: 1D [N].
    Returns [num_segments] or [num_segments, D].
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
    """OCT-DDA official CAT discriminator: 3-d logits in, Linear(3->200)+ReLU x2 + Linear(200,1)+Sigmoid."""

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
    # official transform has no ImageNet Normalize
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch, normalize=False)
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))
    print("  [CAT] OCT-DDA official: vgg16_bn.features(25088) + SNTG/prototype on logits + Π-model teacher")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = VGGBnBackbone().to(device)
    clf = VGGBnClassifier(num_classes=n_cls).to(device)
    netD = Discriminator(input_features=n_cls).to(device)
    print("  [CAT] models ready (feature dim=25088, discriminator on logits)")

    # official hyperparams: lr=0.001; LAMBDA=0.01 (paper says 30)
    LAMBDA = args.margin            # 0.01 (official)
    lamb2_rampup = 5                # sntg weight ramp-up start epoch
    lamb2_rampup_win = 15           # ramp-up window (rises to ~1 by epoch 20)

    optimizer = optim.SGD(list(enc.parameters()) + list(clf.parameters()),
                          lr=args.lr, momentum=0.9)
    # official: discriminator lr fixed at 0.001 (independent of main lr)
    optimizerD = optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    src_train = data['src_train']
    tgt_train = data['tgt_train']
    n_iter = min(len(src_train), len(tgt_train))

    gd = 0
    for epoch in range(args.epochs):
        enc.train(); clf.train(); netD.train()
        gd += 1
        # sntg weight: ramp-up from epoch 5, ~1.0 at epoch 20
        lamb2 = math.exp(-(1.0 - min((gd - lamb2_rampup) * 1.0 / lamb2_rampup_win, 1.0)) * 10.0) \
            if gd >= lamb2_rampup else 0.0
        lamb = adaptation_factor(gd * 1.0 / args.epochs)   # unused (kept as in official code)
        iter_src = iter(src_train)
        iter_tgt = iter(tgt_train)

        for i in range(n_iter):
            xs, ys = next(iter_src)
            xt, _ = next(iter_tgt)
            if xs.size(0) != xt.size(0):
                continue   # equivalent to official drop_last: skip batch when src/tgt sizes differ (onehot/scatter need same shape)
            if torch.cuda.is_available():
                xs, ys, xt = xs.cuda(), ys.cuda(), xt.cuda()
            bs = xs.size(0)

            optimizer.zero_grad()
            optimizerD.zero_grad()

            # -- source classification loss --
            source_logits = clf(enc(xs))                     # [B, n_cls], the paper's "feature space"
            loss_cls = criterion(source_logits, ys)

            # -- target: Pi-model teacher (two forwards, second detached as pseudo-label) --
            target_logits = clf(enc(xt))                     # with grad
            with torch.no_grad():
                target_logits_2 = clf(enc(xt)).detach()      # teacher (dropout perturbation)
            target_pred = target_logits_2.argmax(dim=1)
            target_onehot = torch.zeros(bs, n_cls, device=device).scatter_(1, target_pred.unsqueeze(-1), 1)

            # -- domain discrimination (input = logits) --
            D_src = netD(source_logits)                      # [B, 1]
            D_tgt = netD(target_logits)
            D_real_loss = torch.mean(nn.MultiLabelSoftMarginLoss()(
                D_tgt, torch.ones_like(D_tgt).to(device)))
            D_fake_loss = torch.mean(nn.MultiLabelSoftMarginLoss()(
                D_src, torch.zeros_like(D_src).to(device)))
            # official: D_loss scaled by 0.1 (RevGrad convention); only loss.backward
            # (retain_graph) + D_loss.backward() exist in the official code (no G_loss=-D_loss)
            D_loss = 0.1 * (D_real_loss + D_fake_loss)

            # -- L_c(source): same-class indicator delta from true labels --
            y_onehot = torch.zeros(bs, n_cls, device=device).scatter_(
                1, ys.unsqueeze(-1), 1)
            graph_src = torch.sum(y_onehot[:, None, :] * y_onehot[None, :, :], dim=2)
            dist_src = torch.mean((source_logits[:, None, :] - source_logits[None, :, :]) ** 2, dim=2)
            source_sntg = torch.mean(graph_src * dist_src
                                     + (1 - graph_src) * torch.relu(LAMBDA - dist_src))

            # -- L_c(target): same-class indicator from teacher pseudo-labels --
            graph_tgt = torch.sum(target_onehot[:, None, :] * target_onehot[None, :, :], dim=2)
            dist_tgt = torch.mean((target_logits[:, None, :] - target_logits[None, :, :]) ** 2, dim=2)
            target_sntg = torch.mean(graph_tgt * dist_tgt
                                     + (1 - graph_tgt) * torch.relu(LAMBDA - dist_tgt))

            # -- L_a: class-prototype alignment (fm_mask drops missing classes and normalizes) --
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

            # official: backward main loss and D_loss (with 0.1 scaling) separately
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
                        help='OCT-DDA official default 30; this project early-stops at 15 epochs (collapses after lamb2 ramps up to 1.0, decided by user 2026-08-09)')
    parser.add_argument('--batch', type=int, default=8,
                        help='OCT-DDA official batch=8')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='OCT-DDA official CAT lr=0.001')
    parser.add_argument('--margin', type=float, default=0.01,
                        help='OCT-DDA official LAMBDA=0.01 (comments note paper value 30)')
    parser.add_argument('--input_size', type=int, default=224)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

"""CAN (Contrastive Adaptation Network) baseline.

Kang, Jiang, Yang, Hauptmann, "Contrastive Adaptation Network for
Unsupervised Domain Adaptation", CVPR 2019.
Official repo: https://github.com/thuml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation

Core idea: class-level cross-domain contrastive learning. Pairs whose features
share the same class label are pulled together, pairs with different labels
are pushed apart, so alignment is carried by category semantics rather than by
instance identity (adversarial discrimination).

Adapted to the unified comparison protocol (ResNet-50 backbone, OCT 3-class,
same data split / evaluation as the other baselines):
  - source labels are ground truth; target pseudo-labels come from the
    classifier's argmax (detached), which is the standard UDA reading of CAN
  - loss = CE(source) + lambda * InfoNCE-style contrastive loss over
    L2-normalized features (temperature tau); same-class pairs are positives,
    cross-class pairs are negatives (in-batch)

Usage:
    python -m comparison_experiments.can.train --src A --tgt B --seed 777
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.common.data_loader import load_task, set_seed, LABEL_TO_DATASET
from comparison_experiments.common.evaluate import test
from comparison_experiments.common.models import build_models


def contrastive_loss(fs, ft, ys, yt, tau=0.05):
    """InfoNCE-style class-level contrastive loss over L2-normalized features.

    Concat source + target features; positives = same-class pairs (excluding
    self), negatives = everything else. Temperature tau follows CAN (0.05).
    """
    fs = F.normalize(fs, p=2, dim=1)
    ft = F.normalize(ft, p=2, dim=1)
    feats = torch.cat([fs, ft], dim=0)          # [2B, D]
    labels = torch.cat([ys, yt], dim=0)         # [2B]
    N = feats.size(0)

    sim = feats @ feats.t() / tau               # [2B, 2B]
    pos_mask = (labels[:, None] == labels[None, :]).float()
    pos_mask.fill_diagonal_(0)

    exp_sim = torch.exp(sim)
    pos = (exp_sim * pos_mask).sum(dim=1)       # same-class neighbors
    neg = exp_sim.sum(dim=1) - exp_sim.diag()   # all but self
    loss = -torch.log((pos + 1e-8) / (neg + 1e-8)).mean()
    return loss


def train(args):
    set_seed(args.seed)
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch)
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc, clf = build_models(len(data['class_names']), device)
    # same SGD protocol as MCC / thuml baselines
    optimizer = optim.SGD(clf.get_parameters(base_lr=1.0, backbone=enc),
                          lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer,
                            lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    ce = nn.CrossEntropyLoss()

    src_train = data['src_train']
    tgt_train = data['tgt_train']
    n_iter = min(len(src_train), len(tgt_train))

    for epoch in range(args.epochs):
        enc.train(); clf.train()
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
            s_logits = clf(fs)
            t_logits = clf(ft)
            loss_cls = ce(s_logits, ys)
            # target pseudo-labels (detached) for contrastive alignment
            with torch.no_grad():
                yt_pseudo = t_logits.detach().argmax(dim=1)
            loss_con = contrastive_loss(fs, ft, ys, yt_pseudo, tau=args.tau)
            loss = loss_cls + args.lambda_c * loss_con
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        acc, auc, _ = test(enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                           num_classes=len(data['class_names']), class_names=data['class_names'])
        print("  [CAN] epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f}".format(epoch + 1, args.epochs, acc, auc))


def main():
    parser = argparse.ArgumentParser(description='CAN baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=36)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--tau', type=float, default=0.05,
                        help='contrastive temperature (CAN official 0.05)')
    parser.add_argument('--lambda_c', type=float, default=1.0,
                        help='contrastive loss weight')
    parser.add_argument('--input_size', type=int, default=224)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

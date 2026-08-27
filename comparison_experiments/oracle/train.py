"""Oracle upper bound: fully supervised training on the target domain (no UDA).

Reviewer M3 requested the supervised upper bound with target labels available, to
show FCCAN (no target labels) approaches it; pairs with the source-only lower bound.
Same protocol as UDA: train on target train only (no val merge, pct=50), test on
tgt_test, ResNet-50 + thuml layered lr, 5 seeds.

Usage: python -m comparison_experiments.oracle.train --tgt B --seed 777
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.common.data_loader import load_task, set_seed, LABEL_TO_DATASET
from comparison_experiments.common.evaluate import test
from comparison_experiments.common.models import build_models

# target label -> task name (same as the main table)
TGT_TO_TASK = {'B': 'A->B', 'C': 'A->C'}   # A->B: target B; A->C / B->C: target C
SEEDS = [42, 123, 777, 2024, 3407]


def train_oracle(enc, clf, loader, device, epochs, lr):
    # same thuml ERM protocol as source-only (layered lr + nesterov + wd + LambdaLR)
    if hasattr(clf, 'get_parameters'):
        param_groups = clf.get_parameters(base_lr=1.0, backbone=enc)
    else:
        param_groups = [{'params': list(enc.parameters()) + list(clf.parameters()), 'lr': lr}]
    optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    lr_scheduler = LambdaLR(optimizer,
                            lambda x: lr * (1. + 0.001 * float(x)) ** (-0.75))
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        enc.train(); clf.train()
        for xs, ys in loader:
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            optimizer.zero_grad()
            loss = ce(clf(enc(xs)), ys)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        print("  [Oracle] epoch {}/{} loss={:.4f}".format(ep + 1, epochs, loss.item()))


def run_oracle(tgt_label, seed, epochs=10, lr=0.001, batch=16, input_size=224):
    """Fully supervised training on the target domain; returns (acc, auc).

    Since 2026-08-15: train only on the target train set (pct=50, no val merge);
    test on the same tgt_test.
    """
    set_seed(seed)
    # oracle needs no source; use load_task to reuse splits (src=A is unused)
    data = load_task('A', tgt_label, input_size=input_size,
                     batch_src=batch, batch_tgt=batch,
                     tmi_target_unlabeled_pct=50)  # train set only, no val merge
    n_cls = len(data['class_names'])
    task_name = TGT_TO_TASK.get(tgt_label, '->' + tgt_label)
    print("task={}  oracle(fully supervised target)  target={}  classes={}".format(
        task_name, LABEL_TO_DATASET[tgt_label], data['class_names']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc, clf = build_models(n_cls, device, backbone='resnet50')
    train_oracle(enc, clf, data['tgt_train'], device, epochs=epochs, lr=lr)

    acc, auc, _ = test(enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                       num_classes=n_cls, class_names=data['class_names'])
    print("  [Oracle] final tgt_acc={:.4f} tgt_auc={:.4f}".format(acc, auc))
    return acc, auc


def main():
    parser = argparse.ArgumentParser(description='Oracle upper bound (target supervised)')
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=10,
                        help='Target training epochs (same budget as compared methods)')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--input_size', type=int, default=224)
    args = parser.parse_args()
    run_oracle(args.tgt, args.seed, epochs=args.epochs, lr=args.lr,
               batch=args.batch, input_size=args.input_size)


if __name__ == '__main__':
    main()

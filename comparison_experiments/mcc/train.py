"""MCC (Minimum Class Confusion) baseline.

Jin et al., "Minimum Class Confusion for Versatile Domain Adaptation", ECCV 2020.
Code adapted from thuml/Versatile-Domain-Adaptation for OCT cross-device UDA.

Non-adversarial representative baseline: minimizes class confusion between
source and target via a transport problem over the softmax confusion matrix.
Usage:
    python -m comparison_experiments.mcc.train --src A --tgt B --seed 777
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


def mcc_loss(logits, temperature=2.5):
    """Official MCC loss (Jin et al. ECCV'20, thuml/Versatile-Domain-Adaptation).

    Matches the released code exactly:
        p = softmax(logits / temperature)
        w = 1 + exp(-entropy(p))                    # entropy weighting
        w = batch * w / sum(w)                       # normalized
        cov = (p * w)^T @ p                          # class confusion on target
        cov = cov / sum(cov, dim=1)                  # row-normalize
        loss = (sum(cov) - trace(cov)) / num_classes # off-diagonal / C
    Only the UNLABELED TARGET softmax is used (MCC is a target-only regularizer),
    so there is no cross-domain matmul and no batch-size mismatch issue.
    """
    p = torch.softmax(logits / temperature, dim=1)   # [B, C]
    ent = -(p * torch.log(p + 1e-5)).sum(dim=1)      # [B]
    w = 1.0 + torch.exp(-ent)
    w = p.size(0) * w / (w.sum().detach() + 1e-8)    # [B] normalized
    cov = (p * w.view(-1, 1)).t() @ p                # [C, C]
    cov = cov / cov.sum(dim=1, keepdim=True).clamp_min(1e-8)
    num_classes = p.size(1)
    return (cov.sum() - torch.trace(cov)) / num_classes


def train(args):
    set_seed(args.seed)
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch)
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc, clf = build_models(len(data['class_names']), device)
    # thuml 官方：单一 SGD 分层 lr（backbone 0.1×）+ nesterov + wd + LambdaLR
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
            # official MCC: target-only class-confusion regularizer (Jin ECCV'20)
            loss_mcc = mcc_loss(t_logits, temperature=args.temp)
            loss = loss_cls + args.lambda_mcc * loss_mcc
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        acc, auc, _ = test(enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                           num_classes=len(data['class_names']), class_names=data['class_names'])
        print("  [MCC] epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f}".format(epoch + 1, args.epochs, acc, auc))


def main():
    parser = argparse.ArgumentParser(description='MCC baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=20,
                        help='thuml 官方 MCC epochs=20')
    parser.add_argument('--batch', type=int, default=36,
                        help='thuml 官方 MCC batch=36')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='thuml 官方 MCC lr=0.005')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='thuml 官方 MCC wd=1e-3')
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lambda_mcc', type=float, default=1.0,
                        help='MCC loss weight (thuml 官方 trade_off=1.0)')
    parser.add_argument('--temp', type=float, default=2.5)
    parser.add_argument('--input_size', type=int, default=224,
                        help='thuml ResNet 官方输入 224')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

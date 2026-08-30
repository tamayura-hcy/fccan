"""Oracle upper bound：目标域全监督训练（不做跨域），测目标域测试集。

审稿意见 M3 要求补 oracle 上界行——即"如果目标域全标签可用"的监督学习上界，
用于说明 FCCAN（无目标域标签）已逼近/接近该上界。与 source-only 互为上下界。

与 UDA 协议严格一致：
  - 训练数据 = 目标域训练集（2026-08-15 起：只用 train，不合并 val；
    tmi_target_unlabeled_pct=50）；
  - 测试 = 同一 tgt_test（25% 目标域）；
  - ResNet-50 主干、thuml 分层 lr 协议、5 种子（42/123/777/2024/3407）。

Usage:
    python -m comparison_experiments.oracle.train --tgt B --seed 777
    python -m comparison_experiments.oracle.run_all            # 3 任务 × 5 种子
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

# 目标域标签 → 任务名（与主表场景一致）
TGT_TO_TASK = {'B': 'A->B', 'C': 'A->C'}   # A->B: 目标 B；A->C / B->C: 目标 C
SEEDS = [42, 123, 777, 2024, 3407]


def train_oracle(enc, clf, loader, device, epochs, lr):
    # 与 source-only 相同的 thuml ERM 协议（分层 lr + nesterov + wd + LambdaLR）
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
    """对目标域做全监督训练并返回 (acc, auc)。

    ★ 2026-08-15 用户决策：oracle 训练只用目标域训练集（tmi_target_unlabeled_pct=50，
      不合并 val），测试仍用同一 tgt_test。
    """
    set_seed(seed)
    # oracle 不需要源域；仍走 load_task 以便复用数据划分，src 用 A 即可（不会用到）
    data = load_task('A', tgt_label, input_size=input_size,
                     batch_src=batch, batch_tgt=batch,
                     tmi_target_unlabeled_pct=50)  # 只用训练集，不合并 val
    n_cls = len(data['class_names'])
    task_name = TGT_TO_TASK.get(tgt_label, '->' + tgt_label)
    print("task={}  oracle(全监督目标域)  target={}  classes={}".format(
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
                        help='目标域训练轮数（与对比方法统一预算一致）')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--input_size', type=int, default=224)
    args = parser.parse_args()
    run_oracle(args.tgt, args.seed, epochs=args.epochs, lr=args.lr,
               batch=args.batch, input_size=args.input_size)


if __name__ == '__main__':
    main()

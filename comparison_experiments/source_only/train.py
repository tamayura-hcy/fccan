"""Source-only baseline：源域 CE 训练后直接测目标域（不做任何域适应）。

UDA 对比表的必备基线行，用于证明 DA 方法的有效性。
按 backbone 提供 3 个变体（对应不同方法族）：
  - resnet50  : 对应 DANN/CDAN/DAN/BNM/MCC/SHOT/TENT/SVDNA（thuml ResNet50）
  - vgg16     : 对应 ADDA / EM-DDA（论文用 VGG-16）
  - resnet18  : 对应 EM-DDA/DAGCN 论文里的 ResNet-18 基线

Usage:
    python -m comparison_experiments.source_only.train --src A --tgt B --seed 777 --backbone resnet50
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


def train_source(enc, clf, src_loader, device, epochs, lr):
    # thuml 官方 ERM：分层 lr（backbone 0.1×，bottleneck/head 1×）+ nesterov + wd + LambdaLR
    if hasattr(clf, 'get_parameters'):
        param_groups = clf.get_parameters(base_lr=1.0, backbone=enc)
    else:  # VGG 等无分层的分类头
        param_groups = [{'params': list(enc.parameters()) + list(clf.parameters()), 'lr': lr}]
    optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    lr_scheduler = LambdaLR(optimizer,
                            lambda x: lr * (1. + 0.001 * float(x)) ** (-0.75))
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        enc.train(); clf.train()
        for xs, ys in src_loader:
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            optimizer.zero_grad()
            loss = ce(clf(enc(xs)), ys)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        print("  [Source-only] source epoch {}/{} loss={:.4f}".format(ep + 1, epochs, loss.item()))


def train(args):
    set_seed(args.seed)
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch)
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))
    print("  [Source-only] backbone = {}（无域适应，源域训练后直接测目标域）".format(args.backbone))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc, clf = build_models(n_cls, device, backbone=args.backbone)
    train_source(enc, clf, data['src_train'], device, epochs=args.epochs, lr=args.lr)

    acc, auc, _ = test(enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                       num_classes=n_cls, class_names=data['class_names'])
    print("  [Source-only] final tgt_acc={:.4f} tgt_auc={:.4f}".format(acc, auc))


def main():
    parser = argparse.ArgumentParser(description='Source-only baseline (no adaptation)')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=10,
                        help='源域训练轮数（与对比方法统一预算一致）')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='ResNet/VGG 源域训练 lr（thuml/EM-DDA 论文均 0.001）')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet18', 'vgg16'])
    parser.add_argument('--input_size', type=int, default=224,
                        help='ResNet 官方输入 224（VGG16 强制 224）')
    args = parser.parse_args()
    # VGG-16 官方输入 224（fc 层固定 4096 维）
    if args.backbone == 'vgg16':
        args.input_size = 224
    train(args)


if __name__ == '__main__':
    main()

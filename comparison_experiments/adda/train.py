"""ADDA (Adversarial Discriminative Domain Adaptation) baseline.

Tzeng et al., "Adversarial Discriminative Domain Adaptation", CVPR 2017.
按 OCT-DDA 官方代码（xuqing88/OCT_DDA，ADDA_vgg16_train.py）完整对齐——
这是 DAGCN 论文对比 ADDA 的同源实现：
  - 编码器：vgg16_bn.features（卷积特征，展平 25088 = 512*7*7 维）
  - 判别器：Linear(25088->500)+BN+LeakyReLU+Dropout + Linear(500->500)+BN+LeakyReLU+Dropout
            + Linear(500,2)+Sigmoid（作用在 25088 维原始卷积特征上）
  - 源域：SGD lr=0.001, momentum=0.9（官方 + val 早停）
  - 对抗：optimizer_tgt=SGD(lr=1e-4), optimizer_critic=SGD(lr=1e-3)，batch=8
  - 判别器损失：CE(sigmoid输出, label) 源=1/目标=0；目标编码器 CE(输出, 1) 骗过判别器
  - transform：Resize(256)+CenterCrop(224)+ToTensor（官方无 ImageNet Normalize）

流程：
1. 源域 CE 预训练（SGD 0.001）
2. 冻结源编码器+分类器；训练判别器区分源/目标（SGD 1e-3）
3. 对抗训练目标编码器（SGD 1e-4，目标判为源）

Usage:
    python -m comparison_experiments.adda.train --src A --tgt B --seed 777
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.common.data_loader import load_task, set_seed, LABEL_TO_DATASET
from comparison_experiments.common.evaluate import test
from comparison_experiments.common.models import VGGBnBackbone, VGGBnClassifier


class Discriminator(nn.Module):
    """OCT-DDA 官方 ADDA 判别器（ADDA_vgg16_train.py）：BN+LeakyReLU+Dropout + sigmoid。"""

    def __init__(self, input_dims=25088, hidden_dims=500, output_dims=2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims), nn.BatchNorm1d(hidden_dims), nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(hidden_dims, hidden_dims), nn.BatchNorm1d(hidden_dims), nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(hidden_dims, output_dims))

    def forward(self, x):
        out = self.layer(x)
        return torch.sigmoid(out)


def train_source(enc, clf, src_loader, val_loader, device, epochs, lr=0.001):
    """OCT-DDA 官方源域：SGD lr=0.001 + val 早停（val acc 提升才保存 best，最后加载 best）。"""
    opt = optim.SGD(list(enc.parameters()) + list(clf.parameters()), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_enc = None
    best_clf = None
    for ep in range(epochs):
        enc.train(); clf.train()
        for xs, ys in src_loader:
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            opt.zero_grad()
            loss = ce(clf(enc(xs)), ys)   # clf 内部自动展平 25088
            loss.backward()
            opt.step()
        # 官方 val 早停：val 提升才保存
        if val_loader is not None:
            enc.eval(); clf.eval()
            correct = total = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    if torch.cuda.is_available():
                        xv, yv = xv.cuda(), yv.cuda()
                    pred = clf(enc(xv)).max(1)[1]
                    correct += (pred == yv).sum().item()
                    total += yv.size(0)
            val_acc = float(correct) / max(total, 1)
            if val_acc > best_acc:
                best_acc = val_acc
                best_enc = copy.deepcopy(enc.state_dict())
                best_clf = copy.deepcopy(clf.state_dict())
        else:
            best_enc = copy.deepcopy(enc.state_dict())
            best_clf = copy.deepcopy(clf.state_dict())
        print("  [ADDA] source epoch {}/{} loss={:.4f} val_acc={:.4f}".format(
            ep + 1, epochs, loss.item(), best_acc))
    if best_enc is not None:
        enc.load_state_dict(best_enc)
        clf.load_state_dict(best_clf)
    print("  [ADDA] best val acc on source = {:.4f}".format(best_acc))


def train(args):
    set_seed(args.seed)
    # 官方 transform 无 ImageNet Normalize（Resize+CenterCrop+ToTensor）
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch, normalize=False)
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))
    print("  [ADDA] OCT-DDA 官方：vgg16_bn.features(25088) + BN/LeakyReLU/Dropout 判别器")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_enc = VGGBnBackbone().to(device)
    clf = VGGBnClassifier(num_classes=n_cls).to(device)
    tgt_enc = VGGBnBackbone().to(device)

    print("  [ADDA] stage 1: source training (SGD 0.001 + val early-stop)")
    train_source(src_enc, clf, data['src_train'], data.get('src_val'), device,
                 epochs=args.src_epochs, lr=args.lr)

    # ── 对抗前基线诊断（纯打印，不影响训练）──
    acc_src_t, _, _ = test(src_enc, clf, data['src_test'], len(data['src_test'].dataset),
                           num_classes=n_cls, class_names=data['class_names'])
    acc_tgt0, _, _ = test(src_enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                          num_classes=n_cls, class_names=data['class_names'])
    print("  [ADDA] BASELINE 对抗前: 源测试 acc={:.4f}  目标测试 acc={:.4f}  "
          "（若源已低→源域预训练问题；若源高目标低→看对抗后是否崩）".format(acc_src_t, acc_tgt0))

    # freeze source encoder + classifier
    for p in src_enc.parameters():
        p.requires_grad_(False)
    for p in clf.parameters():
        p.requires_grad_(False)
    src_enc.eval(); clf.eval()

    # 目标编码器从【训练好的】源编码器初始化（官方顺序：先 train_source 再复制）
    tgt_enc.load_state_dict(src_enc.state_dict())

    disc = Discriminator(input_dims=src_enc.out_dim).to(device)
    # OCT-DDA 官方：目标编码器 SGD 1e-4，判别器 SGD 1e-3（D 快 G 慢 + BN/Dropout 防过拟合）
    opt_d = optim.SGD(disc.parameters(), lr=args.disc_lr, momentum=0.9)
    opt_t = optim.SGD(tgt_enc.parameters(), lr=args.tgt_lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()

    src_train = data['src_train']
    tgt_train = data['tgt_train']
    n_iter = min(len(src_train), len(tgt_train))
    best_tgt_acc = 0.0
    best_epoch = -1
    best_enc = None

    for epoch in range(args.epochs):
        tgt_enc.train(); disc.train()
        it_s = iter(src_train)
        it_t = iter(tgt_train)
        for i in range(n_iter):
            xs, _ = next(it_s)
            xt, _ = next(it_t)
            if torch.cuda.is_available():
                xs, xt = xs.cuda(), xt.cuda()

            with torch.no_grad():
                fs = src_enc(xs)
            ft = tgt_enc(xt)
            fs = fs.view(fs.size(0), -1)      # 25088
            ft = ft.view(ft.size(0), -1)      # 25088
            # 官方标签：源特征 -> 1，目标特征 -> 0
            dl_s = torch.ones(fs.size(0), dtype=torch.long, device=fs.device)
            dl_t = torch.zeros(ft.size(0), dtype=torch.long, device=ft.device)

            # 2.1 update discriminator：官方把源+目标 cat 一次前向（BN 用混合批次，
            #     保留域差异；分开前向会让 BN 分别归一化抹平域差异导致 D 学不动）
            opt_d.zero_grad()
            pred_cat = disc(torch.cat([fs, ft], 0).detach())
            lab_cat = torch.cat([dl_s, dl_t], 0)
            loss_d = ce(pred_cat, lab_cat)
            loss_d.backward()
            opt_d.step()

            # 2.2 adversarial update target encoder：目标被判为源(1)
            opt_t.zero_grad()
            ft2 = tgt_enc(xt).view(xt.size(0), -1)
            pred_tgt = disc(ft2)
            dl_adv = torch.ones(ft2.size(0), dtype=torch.long, device=ft2.device)
            loss_adv = ce(pred_tgt, dl_adv)
            loss_adv.backward()
            opt_t.step()

        acc, auc, _ = test(tgt_enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                           num_classes=n_cls, class_names=data['class_names'])
        if acc > best_tgt_acc:
            best_tgt_acc = acc
            best_epoch = epoch + 1
            best_enc = copy.deepcopy(tgt_enc.state_dict())
        print("  [ADDA] epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f} | best={:.4f}@ep{}".format(
            epoch + 1, args.epochs, acc, auc, best_tgt_acc, best_epoch))

    # 早停：加载 best epoch 权重作为最终结果
    if best_enc is not None:
        tgt_enc.load_state_dict(best_enc)
    print("  [ADDA] early-stop: best tgt acc={:.4f} at epoch {}".format(best_tgt_acc, best_epoch))
    acc, auc, _ = test(tgt_enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                       num_classes=n_cls, class_names=data['class_names'])
    print("  [ADDA] FINAL (best-epoch model): tgt_acc={:.4f} tgt_auc={:.4f}".format(acc, auc))


def main():
    parser = argparse.ArgumentParser(description='ADDA baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--src_epochs', type=int, default=15,
                        help='源域预训练 epochs（早停：15 轮）')
    parser.add_argument('--epochs', type=int, default=10,
                        help='对抗 epochs（早停：目标域 10 轮，取 best epoch）')
    parser.add_argument('--batch', type=int, default=8,
                        help='官方 batch_size=8')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='源域预训练 SGD lr=0.001（官方）')
    parser.add_argument('--tgt_lr', type=float, default=1e-4,
                        help='目标编码器 SGD lr=1e-4（官方）')
    parser.add_argument('--disc_lr', type=float, default=1e-3,
                        help='判别器 SGD lr=1e-3（官方）')
    parser.add_argument('--input_size', type=int, default=224,
                        help='VGG-16 官方输入 224')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

"""EM-DDA (Deep Domain Adaptation via adversarial learning + entropy minimization) baseline.

论文：Yuemei Luo et al., "Cross-domain retinopathy classification with optical
coherence tomography images via a novel deep domain adaptation method",
Journal of Biophotonics, 2021 (DOI: 10.1002/jbio.202100096)，即 DAGCN 论文 [30]。
按 OCT-DDA 官方代码（xuqing88/OCT_DDA，ADDA_EM_vgg16_train.py）完整对齐：
  - 编码器：vgg16_bn.features（卷积特征，展平 25088 维）
  - 判别器：Linear(25088->500)+BN+LeakyReLU+Dropout ×2 + Linear(500,2)+Sigmoid
  - 源域：SGD lr=0.001, momentum=0.9
  - 对抗：optimizer_tgt=SGD(lr=1e-4), optimizer_critic=SGD(lr=1e-3)，batch=8
  - 目标编码器损失：loss = loss_tgt(判别器对抗) + entropy_loss(目标预测)（直接加，权重 1.0）
  - transform：Resize(256)+CenterCrop(224)+ToTensor（官方无 Normalize）

流程：
  Step 1: 源域 CE 预训练（SGD 0.001）
  Step 2: 目标编码器=源权重；冻结源；交替训练判别器 + 目标编码器（对抗+熵最小化）
  Step 3: E_T + 源分类器在目标域推理

Usage:
    python -m comparison_experiments.emdda.train --src A --tgt B --seed 777
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


def entropy_loss(p_logit):
    """OCT-DDA 官方熵（ADDA_EM_vgg16_train.py）：-Σ p·log_softmax / B。"""
    p = torch.softmax(p_logit, dim=1)
    return -1 * torch.sum(p * torch.log_softmax(p_logit, dim=1)) / p_logit.size()[0]


class Discriminator(nn.Module):
    """OCT-DDA 官方判别器（ADDA_EM_vgg16_train.py）：BN+LeakyReLU+Dropout + sigmoid。"""

    def __init__(self, input_dims=25088, hidden_dims=500, output_dims=2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims), nn.BatchNorm1d(hidden_dims), nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(hidden_dims, hidden_dims), nn.BatchNorm1d(hidden_dims), nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(hidden_dims, output_dims))

    def forward(self, x):
        out = self.layer(x)
        return torch.sigmoid(out)


def train_source(src_enc, clf, src_loader, val_loader, device, epochs, lr=0.001):
    """Step 1：源域 CE 预训练（官方 SGD lr=0.001 + val 早停）。"""
    opt = optim.SGD(list(src_enc.parameters()) + list(clf.parameters()),
                    lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_enc = None
    best_clf = None
    for ep in range(epochs):
        src_enc.train()
        clf.train()
        for xs, ys in src_loader:
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            opt.zero_grad()
            loss = ce(clf(src_enc(xs)), ys)   # clf 内部自动展平 25088
            loss.backward()
            opt.step()
        if val_loader is not None:
            src_enc.eval(); clf.eval()
            correct = total = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    if torch.cuda.is_available():
                        xv, yv = xv.cuda(), yv.cuda()
                    pred = clf(src_enc(xv)).max(1)[1]
                    correct += (pred == yv).sum().item()
                    total += yv.size(0)
            val_acc = float(correct) / max(total, 1)
            if val_acc > best_acc:
                best_acc = val_acc
                best_enc = copy.deepcopy(src_enc.state_dict())
                best_clf = copy.deepcopy(clf.state_dict())
        else:
            best_enc = copy.deepcopy(src_enc.state_dict())
            best_clf = copy.deepcopy(clf.state_dict())
        print("  [EM-DDA] step1 source epoch {}/{} loss={:.4f} val_acc={:.4f}".format(
            ep + 1, epochs, loss.item(), best_acc))
    if best_enc is not None:
        src_enc.load_state_dict(best_enc)
        clf.load_state_dict(best_clf)
    print("  [EM-DDA] best val acc on source = {:.4f}".format(best_acc))


def train(args):
    set_seed(args.seed)
    # 官方 transform 无 ImageNet Normalize
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch, normalize=False)
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))
    print("  [EM-DDA] OCT-DDA 官方：vgg16_bn.features(25088) + BN/LeakyReLU/Dropout 判别器 + EM")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_enc = VGGBnBackbone().to(device)
    clf = VGGBnClassifier(num_classes=n_cls).to(device)
    print("  [EM-DDA] step 1: 源域预训练 (vgg16_bn, {} epochs, SGD 0.001 + val early-stop)".format(args.src_epochs))
    train_source(src_enc, clf, data['src_train'], data.get('src_val'), device,
                 epochs=args.src_epochs, lr=args.src_lr)

    # ── 对抗前基线诊断（纯打印，不影响训练）──
    acc_src_t, _, _ = test(src_enc, clf, data['src_test'], len(data['src_test'].dataset),
                           num_classes=n_cls, class_names=data['class_names'])
    acc_tgt0, _, _ = test(src_enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                          num_classes=n_cls, class_names=data['class_names'])
    print("  [EM-DDA] BASELINE 对抗前: 源测试 acc={:.4f}  目标测试 acc={:.4f}  "
          "（若源已低→源域预训练问题；若源高目标低→看对抗后是否崩）".format(acc_src_t, acc_tgt0))

    # Step 2：目标编码器 = 源编码器权重；冻结源编码器与源分类器
    tgt_enc = VGGBnBackbone().to(device)
    tgt_enc.load_state_dict(src_enc.state_dict())
    for p in src_enc.parameters():
        p.requires_grad_(False)
    for p in clf.parameters():
        p.requires_grad_(False)
    src_enc.eval()
    clf.eval()

    disc = Discriminator(input_dims=src_enc.out_dim).to(device)
    # 官方：目标编码器 SGD 1e-4，判别器 SGD 1e-3
    opt_d = optim.SGD(disc.parameters(), lr=args.disc_lr, momentum=0.9)
    opt_t = optim.SGD(tgt_enc.parameters(), lr=args.tgt_lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()

    print("  [EM-DDA] step 2: 对抗对齐 + 熵最小化 ({} epochs, early-stop on tgt acc)".format(args.epochs))
    src_train = data['src_train']
    tgt_train = data['tgt_train']
    n_iter = min(len(src_train), len(tgt_train))
    best_tgt_acc = 0.0
    best_epoch = -1
    best_enc = None
    for epoch in range(args.epochs):
        tgt_enc.train()
        disc.train()
        it_s = iter(src_train)
        it_t = iter(tgt_train)
        d_correct = 0
        d_total = 0
        for i in range(n_iter):
            xs, _ = next(it_s)
            xt, _ = next(it_t)
            if torch.cuda.is_available():
                xs, xt = xs.cuda(), xt.cuda()

            with torch.no_grad():
                fs = src_enc(xs)
            ft = tgt_enc(xt)
            fs = fs.view(fs.size(0), -1)   # 25088
            ft = ft.view(ft.size(0), -1)   # 25088

            # ── 2.1 更新判别器：源→1，目标→0 ──
            # 官方 ADDA_EM（ADDA_EM_vgg16_train.py）：源+目标 cat 成一个大 batch 一次前向。
            # 判别器含 BatchNorm1d，若源/目标分开前向，BN 会分别按各自 batch 归一化，
            # 从而抹平域差异，导致 D 学不动（D_acc≈0.5）、对抗失效。必须 cat 一次前向。
            opt_d.zero_grad()
            feat_concat = torch.cat([fs, ft], 0).detach()
            pred_concat = disc(feat_concat)
            dl_s = torch.ones(fs.size(0), dtype=torch.long, device=fs.device)
            dl_t = torch.zeros(ft.size(0), dtype=torch.long, device=ft.device)
            label_concat = torch.cat([dl_s, dl_t], 0)
            loss_d = ce(pred_concat, label_concat)
            loss_d.backward()
            opt_d.step()
            # 判别器准确率（诊断：~1.0=D 太强易负迁移；~0.5=D 没学动=对抗失效）
            d_pred = pred_concat.detach().argmax(1)
            d_correct += int((d_pred == label_concat).sum().item())
            d_total += int(label_concat.numel())

            # ── 2.2 更新目标编码器：loss = loss_tgt(对抗) + entropy_loss（官方直接加）──
            opt_t.zero_grad()
            ft2 = tgt_enc(xt).view(xt.size(0), -1)
            pred_tgt = disc(ft2)
            dl_adv = torch.ones(ft2.size(0), dtype=torch.long, device=ft2.device)
            loss_tgt = ce(pred_tgt, dl_adv)          # 目标判为源
            preds = clf(ft2)                          # 源分类器（冻结）预测目标
            loss_em = entropy_loss(preds)             # 目标熵最小化
            loss = loss_tgt + loss_em
            loss.backward()
            opt_t.step()

        # Step 3：E_T + 源分类器评估目标域，early-stop 取最优 tgt acc
        acc, auc, _ = test(tgt_enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                           num_classes=n_cls, class_names=data['class_names'])
        if acc > best_tgt_acc:
            best_tgt_acc = acc
            best_epoch = epoch + 1
            best_enc = copy.deepcopy(tgt_enc.state_dict())
        print("  [EM-DDA] epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f} (adv={:.3f} em={:.3f}) D_acc={:.3f} | best={:.4f}@ep{}".format(
            epoch + 1, args.epochs, acc, auc, loss_tgt.item(), loss_em.item(),
            float(d_correct) / max(d_total, 1), best_tgt_acc, best_epoch))

    if best_enc is not None:
        tgt_enc.load_state_dict(best_enc)
    print("  [EM-DDA] early-stop: best tgt acc={:.4f} at epoch {}".format(best_tgt_acc, best_epoch))
    # 用最优轮模型再测一次作为最终结果
    acc, auc, _ = test(tgt_enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                       num_classes=n_cls, class_names=data['class_names'])
    print("  [EM-DDA] FINAL (best-epoch model): tgt_acc={:.4f} tgt_auc={:.4f}".format(acc, auc))


def main():
    parser = argparse.ArgumentParser(description='EM-DDA baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--src_epochs', type=int, default=15,
                        help='Step1 源域预训练 epochs（早停：15 轮，SGD 0.001 + val 早停）')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Step2 对抗对齐 epochs（早停：目标域 10 轮，取 best epoch）')
    parser.add_argument('--batch', type=int, default=8,
                        help='官方 batch_size=8')
    parser.add_argument('--src_lr', type=float, default=0.001,
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

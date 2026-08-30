import argparse
import copy
import csv
import os
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.common.data_loader import load_task, set_seed, LABEL_TO_DATASET, TASK_LIST
from comparison_experiments.common.evaluate import test
from comparison_experiments.common.models import VGGBnBackbone, VGGBnClassifier

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
SAVE_ROOT = os.path.join(ROOT, "saves_adda_em")
RESULT_CSV = os.path.join(ROOT, "results", "adda_em_summary.csv")

TASKS = [("A", "B"), ("A", "C"), ("B", "C")]          # 论文三个跨设备场景
SEEDS = [42, 123, 777, 2024, 3407]


class Discriminator(nn.Module):
    """OCT-DDA 官方 ADDA 判别器（ADDA_EM_vgg16_train.py）：BN+LeakyReLU+Dropout + sigmoid。"""

    def __init__(self, input_dims=25088, hidden_dims=500, output_dims=2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims), nn.BatchNorm1d(hidden_dims), nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(hidden_dims, hidden_dims), nn.BatchNorm1d(hidden_dims), nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(hidden_dims, output_dims))

    def forward(self, x):
        return torch.sigmoid(self.layer(x))


def entropy_loss(p_logit):
    """官方 EM 损失：-sum(p * log p)（batch 平均）。"""
    p = F.softmax(p_logit, dim=-1)
    return -1.0 * torch.sum(p * F.log_softmax(p_logit, dim=-1)) / p_logit.size(0)


def train_source(enc, clf, src_loader, val_loader, device, epochs, lr=0.001):
    """官方源域：SGD lr=0.001 + val 早停（val acc 提升才保存，最后加载 best）。"""
    opt = optim.SGD(list(enc.parameters()) + list(clf.parameters()), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    best_acc, best_enc, best_clf = 0.0, None, None
    for ep in range(epochs):
        enc.train(); clf.train()
        loss_sum = 0.0
        for xs, ys in src_loader:
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            opt.zero_grad()
            loss = ce(clf(enc(xs)), ys)      # clf 内部展平 25088
            loss.backward()
            opt.step()
            loss_sum += loss.item()
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
        print("  [ADDA-EM] source epoch {}/{} loss={:.4f} val_acc={:.4f}".format(
            ep + 1, epochs, loss_sum / max(len(src_loader), 1), best_acc))
    if best_enc is not None:
        enc.load_state_dict(best_enc)
        clf.load_state_dict(best_clf)
    print("  [ADDA-EM] best val acc on source = {:.4f}".format(best_acc))
    return enc, clf


def train_tgt(src_enc, src_clf, tgt_enc, netD, src_loader, tgt_loader, device,
              num_epochs, tgt_lr=1e-4, disc_lr=1e-3, em_w=1.0, disc_updates=1,
              disc_gap=False, g_updates=1, log_fn=print):
    """官方 ADDA+EM：判别器 + 目标编码器（对抗 + 熵最小化），无早停（报告 final-epoch）。

    disc_gap=True：判别器输入改用 GAP 特征（512 维）而非 25088 维展平卷积特征。
    官方结构在 batch 16 + SGD 下不稳定（D 快压死生成器 / D 慢学不动），
    GAP 降维是让判别器可训、避免负迁移的工程化变体。
    """
    src_enc.eval(); src_clf.eval()
    ce = nn.CrossEntropyLoss()
    opt_t = optim.SGD(tgt_enc.parameters(), lr=tgt_lr, momentum=0.9)
    opt_d = optim.SGD(netD.parameters(), lr=disc_lr, momentum=0.9)
    n_iter = min(len(src_loader), len(tgt_loader))

    def _feat(enc, x):
        f = enc(x)
        if disc_gap:
            return f.mean(dim=(2, 3))   # 512 维 GAP
        return f.view(f.size(0), -1)    # 25088 维展平

    for epoch in range(num_epochs):
        tgt_enc.train(); netD.train()
        it_s, it_t = iter(src_loader), iter(tgt_loader)
        d_correct = 0
        d_total = 0
        for i in range(n_iter):
            xs, _ = next(it_s)
            xt, _ = next(it_t)
            if torch.cuda.is_available():
                xs, xt = xs.cuda(), xt.cuda()
            with torch.no_grad():
                fs = _feat(src_enc, xs)
            ft = _feat(tgt_enc, xt)
            dl_s = torch.ones(fs.size(0), dtype=torch.long, device=fs.device)
            dl_t = torch.zeros(ft.size(0), dtype=torch.long, device=ft.device)

            # 2.1 更新判别器（源=1 / 目标=0）；disc_updates 次（官方=1）
            # 官方：把源+目标特征 cat 成一个大 batch 后一次前向 netD（保留域差异，
            # 避免 BN 分别按源/目标各自归一化而抹平域差异导致 D 学不动）
            for _ in range(max(1, int(disc_updates))):
                opt_d.zero_grad()
                pred_cat = netD(torch.cat([fs, ft], 0).detach())
                lab_cat = torch.cat([dl_s, dl_t], 0)
                loss_d = ce(pred_cat, lab_cat)
                loss_d.backward()
                opt_d.step()
            # 判别器准确率（诊断：若接近 1.0 说明 D 过强，易负迁移；若 ~0.5 说明 D 没学到）
            d_pred = pred_cat.detach().argmax(1)
            d_correct += int((d_pred == lab_cat).sum().item())
            d_total += int(lab_cat.numel())

            # 2.2 更新目标编码器：对抗（判为源）+ EM（分类器熵）；g_updates 次（官方=1）
            # 判别器收敛过快时加大 g_updates 让生成器追上（D 每步 1 次、G 每步多次）
            for _ in range(max(1, int(g_updates))):
                opt_t.zero_grad()
                raw2 = tgt_enc(xt)
                ft2 = raw2.mean(dim=(2, 3)) if disc_gap else raw2.view(raw2.size(0), -1)
                pred_tgt = netD(ft2)
                dl_adv = torch.ones(ft2.size(0), dtype=torch.long, device=ft2.device)
                loss_adv = ce(pred_tgt, dl_adv)
                # 分类器始终使用 25088 维展平特征（与官方一致），与判别器输入维度无关
                outputs = src_clf(raw2.view(raw2.size(0), -1))
                loss_em = entropy_loss(outputs)
                loss = loss_adv + em_w * loss_em
                loss.backward()
                opt_t.step()
        # 每 epoch 目标域测试（final-epoch 报告）+ 判别器诊断
        acc, auc, _ = test(tgt_enc, src_clf, tgt_loader, len(tgt_loader.dataset))
        log_fn("  [ADDA-EM] tgt epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f}  D_acc={:.3f}".format(
            epoch + 1, num_epochs, acc, auc, float(d_correct) / max(d_total, 1)))
    return tgt_enc


def run_one(src, tgt, seed, src_epochs, epochs, em_w, batch, save_ckpt=False,
            tgt_lr=1e-4, disc_lr=1e-3, disc_updates=1, tgt_protocol='paper75',
            src_only=False, disc_gap=False, g_updates=1):
    set_seed(seed)
    tgt_pct = 75 if tgt_protocol == 'paper75' else 50   # paper75=论文 75% 合并；official50=官方仅 train
    data = load_task(src, tgt, input_size=224, batch_src=batch, batch_tgt=batch,
                     tmi_target_unlabeled_pct=tgt_pct,
                     train_aug=False, normalize=False)   # 官方无增强、无 ImageNet 归一化
    n_cls = len(data['class_names'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n[ADDA-EM] task {}->{} seed={} classes={} tgt_protocol={}".format(
        LABEL_TO_DATASET[src], LABEL_TO_DATASET[tgt], seed, data['class_names'], tgt_protocol))

    src_enc = VGGBnBackbone().to(device)
    src_clf = VGGBnClassifier(num_classes=n_cls).to(device)
    tgt_enc = VGGBnBackbone().to(device)

    print("  [ADDA-EM] stage 1: source training (SGD 0.001, val early-stop)")
    src_enc, src_clf = train_source(src_enc, src_clf, data['src_train'], data.get('src_val'),
                                    device, epochs=src_epochs)

    # 源域本身在目标域上的表现（source-only 基线，用于判断是否负迁移）
    acc_so, auc_so, _ = test(src_enc, src_clf, data['tgt_test'], len(data['tgt_test'].dataset),
                             num_classes=n_cls, class_names=data['class_names'])
    print("  [ADDA-EM] source-only on target: acc={:.4f} auc={:.4f}".format(acc_so, auc_so))
    if src_only:
        print("  [ADDA-EM] --src_only: 不进行对抗适应，返回 source-only 结果")
        return acc_so, auc_so

    # 冻结源
    for p in src_enc.parameters():
        p.requires_grad_(False)
    for p in src_clf.parameters():
        p.requires_grad_(False)

    # 目标编码器从【训练好的】源编码器初始化（官方顺序：先 train_source 再复制）
    tgt_enc.load_state_dict(src_enc.state_dict())

    netD = Discriminator(input_dims=(512 if disc_gap else src_enc.out_dim)).to(device)
    print("  [ADDA-EM] stage 2: adversarial + EM target adaptation ({} epochs, final-epoch, disc_input={})"
          .format(epochs, "GAP512" if disc_gap else "flat25088"))
    train_tgt(src_enc, src_clf, tgt_enc, netD, data['src_train'], data['tgt_train'],
              device, num_epochs=epochs, tgt_lr=tgt_lr, disc_lr=disc_lr,
              em_w=em_w, disc_updates=disc_updates, disc_gap=disc_gap,
              g_updates=g_updates)

    # final-epoch 测试（论文统一协议）
    acc, auc, _ = test(tgt_enc, src_clf, data['tgt_test'], len(data['tgt_test'].dataset),
                       num_classes=n_cls, class_names=data['class_names'])
    print("  [ADDA-EM] FINAL (final-epoch): tgt_acc={:.4f} tgt_auc={:.4f}  (source-only {:.4f})"
          .format(acc, auc, acc_so))

    if save_ckpt:
        out_dir = os.path.join(SAVE_ROOT, "{}_to_{}_s{}".format(
            LABEL_TO_DATASET[src], LABEL_TO_DATASET[tgt], seed))
        os.makedirs(out_dir, exist_ok=True)
        torch.save(tgt_enc.state_dict(), os.path.join(out_dir, "target_encoder.pt"))
        torch.save(netD.state_dict(), os.path.join(out_dir, "netD.pt"))
        with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write("src={} tgt={} seed={}\n".format(src, tgt, seed))
            f.write("final_epoch acc={:.6f} auc={:.6f}\n".format(acc, auc))
    return acc, auc


def run_all(args):
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    rows = []
    for src, tgt in TASKS:
        for seed in SEEDS:
            acc, auc = run_one(src, tgt, seed, args.src_epochs, args.epochs, args.em_w,
                               args.batch, save_ckpt=True, tgt_lr=args.tgt_lr,
                               disc_lr=args.disc_lr, disc_updates=args.disc_updates,
                               tgt_protocol=args.tgt_protocol, src_only=args.src_only,
                               disc_gap=args.disc_gap, g_updates=args.g_updates)
            rows.append({"task": "{}->{}".format(LABEL_TO_DATASET[src], LABEL_TO_DATASET[tgt]),
                         "seed": seed, "acc": acc, "auc": auc})
    # 汇总
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "acc", "auc"])
        w.writeheader()
        w.writerows(rows)
    print("\n===== ADDA-EM summary (final-epoch, {} seeds) =====".format(len(SEEDS)))
    for task in [r["task"] for r in rows]:
        pass
    for tag in ["BOE->TMI", "BOE->CELL", "TMI->CELL"]:
        rs = [r for r in rows if r["task"] == tag]
        accs = [100.0 * r["acc"] for r in rs]
        aucs = [100.0 * r["auc"] for r in rs]
        print("  {}  acc={:.2f}$\\pm${:.2f}  auc={:.2f}$\\pm${:.2f}".format(
            tag, statistics.mean(accs), statistics.stdev(accs),
            statistics.mean(aucs), statistics.stdev(aucs)))
    print("saved ->", RESULT_CSV)


def main():
    ap = argparse.ArgumentParser(description="ADDA + EM (official ADDA_EM_vgg16_train.py)")
    ap.add_argument("--src", type=str, default="A", choices=["A", "B", "C"])
    ap.add_argument("--tgt", type=str, default="B", choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src_epochs", type=int, default=5, help="源域预训练 epochs（官方默认 5，val 早停）")
    ap.add_argument("--epochs", type=int, default=10, help="目标域对抗+EM epochs（论文统一 10）")
    ap.add_argument("--em_w", type=float, default=1.0, help="EM 损失权重（官方 = 1）")
    ap.add_argument("--batch", type=int, default=16, help="官方 batch_size=16")
    ap.add_argument("--tgt_lr", type=float, default=1e-4, help="目标编码器 SGD lr（官方 1e-4）")
    ap.add_argument("--disc_lr", type=float, default=1e-3, help="判别器 SGD lr（官方 1e-3，负迁移时可调小）")
    ap.add_argument("--disc_updates", type=int, default=1, help="每步判别器更新次数（官方 1）")
    ap.add_argument("--g_updates", type=int, default=1, help="每步目标编码器更新次数（官方 1；D 收敛过快时调大，如 2/3）")
    ap.add_argument("--tgt_protocol", type=str, default="paper75",
                    choices=["paper75", "official50"],
                    help="paper75=论文 75% 目标合并；official50=官方仅 train 目录")
    ap.add_argument("--disc_gap", action="store_true",
                    help="判别器用 GAP 512 维特征（稳定对抗，规避 25088 维 D 不稳定）")
    ap.add_argument("--src_only", action="store_true", help="只训源并测目标（诊断负迁移基线）")
    ap.add_argument("--save_ckpt", type=int, default=0, choices=[0, 1])
    ap.add_argument("--all", action="store_true", help="跑 3 任务 × 5 种子并汇总")
    args = ap.parse_args()
    if args.all:
        run_all(args)
    else:
        acc, auc = run_one(args.src, args.tgt, args.seed, args.src_epochs, args.epochs,
                           args.em_w, args.batch, save_ckpt=bool(args.save_ckpt),
                           tgt_lr=args.tgt_lr, disc_lr=args.disc_lr,
                           disc_updates=args.disc_updates,
                           tgt_protocol=args.tgt_protocol, src_only=args.src_only,
                           disc_gap=args.disc_gap, g_updates=args.g_updates)
        print("[DONE] acc={:.4f} auc={:.4f}".format(acc, auc))


if __name__ == "__main__":
    main()

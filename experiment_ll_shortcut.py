#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LL shortcut hypothesis verification.

Trains two source-only models (baseline vs LL-augmented P070A20), then tests
robustness on the target test set under perturbations: original, LL x k
(brightness+contrast), LL_AC x k (contrast only, keeps DC), HF x k (control).
Expected under the hypothesis: the baseline collapses on LL perturbations but not
HF; the augmented model stays robust to LL.

Usage:
    python experiment_ll_shortcut.py --task A-C --seed 777 --src_epochs 8 --skip-train
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torchvision import datasets, transforms

# Reuse main project components
from models.fea_net import FEANet, Classifier
from trainers.source_trainer import train_src
from util.data_utils import get_data_dir, CELL_DIR, LABEL_TO_DATASET, DATASET_TO_LABEL
from util.ll_strength_aug import haar_dwt2d, haar_idwt2d_fixed

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

TASK_LABEL = {"A-B": ("BOE", "TMI"), "A-C": ("BOE", "CELL"), "B-C": ("TMI", "CELL")}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(n_cls, device):
    """FEA-Net main model (params match main.py argparse defaults)."""
    enc = FEANet(wrb_alpha=0.4, wrb_lambda=0.3, use_hf_comp=True, hf_comp_scale=0.2,
                 use_msw_sa=True, msw_sa_positions='3', use_wrb_after_layer2=True).to(device)
    clf = Classifier(enc.combined_features, n_cls, prob=0.3).to(device)
    return enc, clf


def load_task(src_domain, tgt_domain, input_size=224):
    """Load source train/val + target test (same protocol as main.py)."""
    resize_size = int(input_size * 256 / 224)
    tr_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    tr_eval = transforms.Compose([
        transforms.Resize(resize_size), transforms.CenterCrop(input_size), transforms.ToTensor()])

    src_dir = get_data_dir(src_domain, use_bg_removed=True)
    tgt_dir = get_data_dir(tgt_domain, use_bg_removed=True)

    ds_s_train = datasets.ImageFolder(os.path.join(src_dir, 'train'), transform=tr_train)
    ds_s_val = datasets.ImageFolder(os.path.join(src_dir, 'val'), transform=tr_eval)
    ds_t_test = datasets.ImageFolder(os.path.join(tgt_dir, 'test'), transform=tr_eval)
    n_cls = len(ds_s_train.classes)
    assert len(ds_t_test.classes) == n_cls

    dl_s_train = torch.utils.data.DataLoader(ds_s_train, batch_size=32, shuffle=True, num_workers=0)
    dl_s_val = torch.utils.data.DataLoader(ds_s_val, batch_size=32, shuffle=False, num_workers=0)
    dl_t_test = torch.utils.data.DataLoader(ds_t_test, batch_size=32, shuffle=False, num_workers=0)
    return dl_s_train, dl_s_val, dl_t_test, n_cls


def train_src_model(enc, clf, dl_train, dl_val, epochs, save_name, use_ll_aug, ll_alpha, ll_prob, device):
    """Train a source model with the main train_src; return (enc, clf)."""
    print("  [train] {}  src_ll_aug={} alpha={} prob={} epochs={} save={}".format(
        "aug model" if use_ll_aug else "baseline model", float(use_ll_aug), ll_alpha, ll_prob, epochs, save_name))
    # No class_weights: same protocol for a fair comparison.
    enc, clf, _ = train_src(
        enc, clf, dl_train, dl_val, epochs, save_name,
        weight_decay=5e-4, src_optimizer='sgd', base_lr=None,
        src_ce_temperature=5.0, class_weights=None, grad_clip_norm=0.0,
        src_ll_aug=float(use_ll_aug), src_ll_alpha=ll_alpha, src_ll_prob=ll_prob,
        src_swa=0.0, save_weights=0)
    return enc, clf


@torch.no_grad()
def evaluate_acc(enc, clf, loader, device, mode='none', k=1.0):
    """Evaluate acc on loader. mode: 'none' original; 'll_full' LL x k;
    'll_ac' only LL_AC x k (keeps DC); 'hf' LH/HL/HH x k.
    """
    enc.eval()
    clf.eval()
    correct = total = 0
    for x, y in loader:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        if mode != 'none':
            sub = haar_dwt2d(x)                       # (B, 4, C, H/2, W/2)
            if mode == 'll_full':
                sub[:, 0] = sub[:, 0] * k
            elif mode == 'll_ac':
                LL = sub[:, 0]
                mu = LL.mean(dim=(2, 3), keepdim=True)
                sub[:, 0] = mu + k * (LL - mu)
            elif mode == 'hf':
                sub[:, 1:] = sub[:, 1:] * k
            x = haar_idwt2d_fixed(sub).clamp(0.0, 1.0)
        feat = enc(x)[0]
        pred = clf(feat)[0].max(1)[1]
        correct += (pred == y).sum().item()
        total += y.size(0)
    return float(correct) / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description="LL shortcut verification: baseline vs LL-aug on target-test perturbation robustness")
    ap.add_argument("--task", type=str, default="A-B", choices=["A-B", "A-C", "B-C"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src_epochs", type=int, default=10)
    ap.add_argument("--ll_alpha", type=float, default=2.0, help="LL perturbation magnitude of the aug model (2.0 in P070A20)")
    ap.add_argument("--ll_prob", type=float, default=0.7, help="LL perturbation probability of the aug model (0.7 in P070A20)")
    ap.add_argument("--k_values", type=str, default="2,3,4", help="perturbation strength k list (comma-separated)")
    ap.add_argument("--save_root", type=str, default="saves_ll_shortcut")
    ap.add_argument("--skip-train", action="store_true", help="reuse saved weights; only run perturbation tests")
    args = ap.parse_args()

    src_domain, tgt_domain = TASK_LABEL[args.task]
    set_seed(args.seed)
    # CELL split same as main.py
    import util.data_utils as _du
    _du.CELL_DIR = CELL_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[LL-shortcut] task={}({}->{}) seed={} src_epochs={} device={}".format(
        args.task, src_domain, tgt_domain, args.seed, args.src_epochs, device))
    print("  LL aug config: alpha={} prob={} (P070A20)".format(args.ll_alpha, args.ll_prob))

    dl_s_train, dl_s_val, dl_t_test, n_cls = load_task(src_domain, tgt_domain)
    print("  source train={} val={}  target test={}  classes={}".format(
        len(dl_s_train.dataset), len(dl_s_val.dataset), len(dl_t_test.dataset), n_cls))

    base_dir = os.path.join(args.save_root, "{}_s{}".format(args.task, args.seed), "baseline")
    aug_dir = os.path.join(args.save_root, "{}_s{}".format(args.task, args.seed), "aug")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(aug_dir, exist_ok=True)

    # ---- 1) Train the two models (or reuse) ----
    enc_b, clf_b, enc_a, clf_a = None, None, None, None
    if not args.skip_train:
        enc_b, clf_b = build_models(n_cls, device)
        enc_b, clf_b = train_src_model(enc_b, clf_b, dl_s_train, dl_s_val, args.src_epochs,
                                       base_dir, use_ll_aug=0, ll_alpha=0.5, ll_prob=0.5, device=device)
        torch.save(enc_b.state_dict(), os.path.join(base_dir, "enc.pt"))
        torch.save(clf_b.state_dict(), os.path.join(base_dir, "clf.pt"))

        enc_a, clf_a = build_models(n_cls, device)
        enc_a, clf_a = train_src_model(enc_a, clf_a, dl_s_train, dl_s_val, args.src_epochs,
                                       aug_dir, use_ll_aug=1, ll_alpha=args.ll_alpha,
                                       ll_prob=args.ll_prob, device=device)
        torch.save(enc_a.state_dict(), os.path.join(aug_dir, "enc.pt"))
        torch.save(clf_a.state_dict(), os.path.join(aug_dir, "clf.pt"))
    else:
        enc_b, clf_b = build_models(n_cls, device)
        enc_b.load_state_dict(torch.load(os.path.join(base_dir, "enc.pt"), map_location=device))
        clf_b.load_state_dict(torch.load(os.path.join(base_dir, "clf.pt"), map_location=device))
        enc_a, clf_a = build_models(n_cls, device)
        enc_a.load_state_dict(torch.load(os.path.join(aug_dir, "enc.pt"), map_location=device))
        clf_a.load_state_dict(torch.load(os.path.join(aug_dir, "clf.pt"), map_location=device))
        print("  [load] reuse saved weights")

    # ---- 2) Perturbation tests on the target test set ----
    k_vals = [float(v) for v in args.k_values.split(",") if v.strip()]
    modes = [
        ("original", "none", 1.0),
        ("LL-full(bright+contrast)", "ll_full", 0.0),
        ("LL-AC-only(contrast)", "ll_ac", 0.0),
        ("HF(LH/HL/HH control)", "hf", 0.0),
    ]
    print("\n" + "=" * 84)
    print("target-test perturbation robustness  task={} ({}->{})  seed={}".format(
        args.task, src_domain, tgt_domain, args.seed))
    print("{:<28}{:>6}{:>12}{:>12}{:>14}".format("perturbation", "k", "base acc", "aug acc", "diff(aug-base)"))
    results = []
    acc_orig_b = acc_orig_a = None
    for mname, m, _ in modes:
        for k in ([1.0] if m == "none" else k_vals):
            t0 = time.time()
            acc_b = evaluate_acc(enc_b, clf_b, dl_t_test, device, mode=m, k=k)
            acc_a = evaluate_acc(enc_a, clf_a, dl_t_test, device, mode=m, k=k)
            diff = acc_a - acc_b
            if m == "none":
                acc_orig_b, acc_orig_a = acc_b, acc_a
            print("{:<28}{:>6.1f}{:>12.4f}{:>12.4f}{:>14.4f}   {:.0f}s".format(
                mname, k, acc_b, acc_a, diff, time.time() - t0))
            results.append((mname, k, acc_b, acc_a, diff))
    # Drop relative to the original (more intuitive)
    print("\n--- accuracy drop vs original ---")
    print("{:<28}{:>6}{:>14}{:>14}".format("perturbation", "k", "base drop", "aug drop"))
    for mname, m, _ in modes:
        if m == "none":
            continue
        for k in k_vals:
            r = [x for x in results if x[0] == mname and x[1] == k][0]
            print("{:<28}{:>6.1f}{:>14.4f}{:>14.4f}".format(
                mname, k, acc_orig_b - r[2], acc_orig_a - r[3]))

    # Write the results file
    out = os.path.join(ROOT, "results_ll_shortcut.txt")
    with open(out, "a", encoding="utf-8") as f:
        f.write("\n# LL-shortcut {} ({}->{}) seed={} src_epochs={} ll(P{}A{}) {} | original acc: base={:.4f} aug={:.4f}\n".format(
            args.task, src_domain, tgt_domain, args.seed, args.src_epochs,
            int(args.ll_prob * 100), int(args.ll_alpha * 10),
            time.strftime("%Y-%m-%d %H:%M:%S"), acc_orig_b, acc_orig_a))
        for mname, k, ab, aa, d in results:
            f.write("  {:<28} k={:<4.1f}  base={:.4f}  aug={:.4f}  diff={:+.4f}\n".format(mname, k, ab, aa, d))
    print("\nsaved -> {}".format(out))


if __name__ == "__main__":
    main()

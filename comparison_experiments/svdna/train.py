"""SVDNA (Singular Value Decomposition Noise Adaptation) baseline.

Koch et al., "Noise transfer for unsupervised domain adaptation of retinal OCT
images", MICCAI 2022. Official repo: https://github.com/ValentinKoch/SVDNA

OCT-specific, non-adversarial representative: restyle source images to match
the target domain's noise / pixel-intensity statistics via SVD, then train a
classifier on the restyled source images.

SVDNA algorithm (per paper):
  For each source image, decompose with SVD, rebuild it using target-domain
  singular values / pixel statistics so the noise pattern resembles target.
  We approximate per-image: normalize source intensity to target mean/std
  (pixel-intensity match) + apply SVD low-rank noise transfer.

Usage:
    python -m comparison_experiments.svdna.train --src A --tgt B --seed 777
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.common.data_loader import load_task, set_seed, LABEL_TO_DATASET
from comparison_experiments.common.evaluate import test
from comparison_experiments.common.models import build_models
from torchvision import datasets, transforms


def svdna_restyle(src_tensor, tgt_spectra, alpha=0.8):
    """Restyle a source batch toward target noise statistics via SVD spectrum transfer.

    src_tensor: [B, C, H, W] in [0,1].
    Paper (Koch MICCAI'22) core: decompose each source image with SVD and rebuild
    it using the *target-domain* singular-value spectrum, so the source image
    carries target-domain noise / pixel-intensity statistics. We compute the mean
    target spectrum once (per channel) and mix it (energy-scaled) into each
    source image's singular values.
    """
    B = src_tensor.size(0)
    out = src_tensor.clone()
    for i in range(B):
        img = out[i]  # [C,H,W]
        for c in range(img.size(0)):
            ts = tgt_spectra[c]
            if ts is None:
                continue
            m = img[c]
            try:
                U, S, Vt = torch.linalg.svd(m, full_matrices=False)
                ts = ts.to(S.device, S.dtype)
                # scale target spectrum to source total energy, then mix
                scale = S.sum() / (ts.sum() + 1e-8)
                S_new = (1 - alpha) * S + alpha * (ts * scale)
                img[c] = (U * S_new) @ Vt
            except Exception:
                pass
    return out


def compute_target_svd_spectrum(tgt_loader, n_ref=24):
    """Sample n_ref target images and compute per-channel mean singular-value spectrum.

    This is the target-domain noise profile used by svdna_restyle (the SVDNA
    'noise statistics of the target domain').
    """
    spectra = [[] for _ in range(3)]
    count = 0
    with torch.no_grad():
        for xt, _ in tgt_loader:
            for i in range(xt.size(0)):
                img = xt[i]
                for c in range(img.size(0)):
                    try:
                        _, S, _ = torch.linalg.svd(img[c].float(), full_matrices=False)
                        spectra[c].append(S)
                    except Exception:
                        pass
                count += 1
                if count >= n_ref:
                    break
            if count >= n_ref:
                break
    return [torch.stack(v).mean(0) if v else None for v in spectra]


def train(args):
    set_seed(args.seed)
    # 与其他对比方法统一：用 load_task 加载（含 CELL 75% 协议、按人划分等）
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch)
    dl_src = data['src_train']
    dl_tgt_stats = data['tgt_train']   # 目标域训练集：算 SVD 噪声频谱
    dl_tgt_test = data['tgt_test']
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc, clf = build_models(n_cls, device)
    opt = optim.SGD(list(enc.parameters()) + list(clf.parameters()), lr=args.lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()

    # target noise spectrum (computed once from sampled target images)
    tgt_spectra = compute_target_svd_spectrum(dl_tgt_stats)
    print("  [SVDNA] target spectrum length per channel: {}".format(
        [s.size(0) if s is not None else None for s in tgt_spectra]))

    for epoch in range(args.epochs):
        enc.train(); clf.train()
        for xs, ys in dl_src:
            xs = svdna_restyle(xs, tgt_spectra, alpha=args.alpha)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            opt.zero_grad()
            loss = ce(clf(enc(xs)), ys)
            loss.backward()
            opt.step()

        acc, auc, _ = test(enc, clf, dl_tgt_test, len(dl_tgt_test.dataset),
                           num_classes=n_cls, class_names=data['class_names'])
        print("  [SVDNA] epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f}".format(epoch + 1, args.epochs, acc, auc))


def main():
    parser = argparse.ArgumentParser(description='SVDNA baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='SVDNA spectrum-mixing ratio: how much target spectrum to inject (official-style noise transfer)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='thuml ResNet 官方输入 224')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

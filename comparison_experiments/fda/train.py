"""FDA (Fourier Domain Adaptation) baseline.

Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation",
CVPR 2020.  Official repo: https://github.com/YanchaoYang/FDA

Core idea: before supervised training on the source, swap the low-frequency
amplitude of source images with that of target images (keeping the source
phase), so the model sees target-style low-frequency statistics and has to
rely on phase/edge information that transfers across devices.

Adapted to the unified comparison protocol (ResNet-50 backbone, OCT 3-class,
same data split / evaluation as the other baselines):
  - per step, take one source batch and one target batch
  - apply the FDA low-frequency amplitude swap to the source batch (beta=0.01)
  - train with plain cross-entropy on the swapped source batch

Usage:
    python -m comparison_experiments.fda.train --src A --tgt B --seed 777
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


def fda_swap_amplitude(src, tgt, beta=0.01):
    """Official FDA low-frequency amplitude swap (torch, batch-safe).

    Matches the released implementation (low_freq_mutate_np):
      b = sqrt(beta) * H, c = sqrt(beta) * W  (full box size),
      the centered box of the source amplitude is replaced by the target's.
    Works directly on the normalized tensors because the Fourier transform is
    linear and only the amplitude is swapped (phase preserved).

    The target batch may be smaller than the source batch on the last step
    (e.g. 12 vs 36); the official code iterates per image and cycles the
    target index (tgt[i % tgt.size(0)]).  We do the same with vectorized
    index cycling so it works for any batch sizes.
    """
    fft_src = torch.fft.fft2(src, dim=(-2, -1))
    fft_tgt = torch.fft.fft2(tgt, dim=(-2, -1))
    amp_src = torch.abs(fft_src)
    pha_src = torch.angle(fft_src)
    amp_tgt = torch.abs(fft_tgt)

    _, _, h, w = amp_src.shape
    bh = max(1, int(torch.floor(torch.sqrt(torch.tensor(beta)) * h).item()))
    bw = max(1, int(torch.floor(torch.sqrt(torch.tensor(beta)) * w).item()))
    h1, h2 = h // 2 - bh // 2, h // 2 + bh // 2
    w1, w2 = w // 2 - bw // 2, w // 2 + bw // 2

    n, nt = amp_src.size(0), amp_tgt.size(0)
    idx = torch.arange(n, device=amp_src.device) % nt   # cycle target images
    amp_src[:, :, h1:h2, w1:w2] = amp_tgt[idx][:, :, h1:h2, w1:w2]

    fft_new = amp_src * torch.exp(1j * pha_src)
    return torch.fft.ifft2(fft_new, dim=(-2, -1)).real


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

            # FDA: swap low-frequency amplitude of source with target, keep phase
            xs_ = fda_swap_amplitude(xs, xt, beta=args.beta)

            optimizer.zero_grad()
            logits = clf(enc(xs_))
            loss = ce(logits, ys)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        acc, auc, _ = test(enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                           num_classes=len(data['class_names']), class_names=data['class_names'])
        print("  [FDA] epoch {}/{} tgt_acc={:.4f} tgt_auc={:.4f}".format(epoch + 1, args.epochs, acc, auc))


def main():
    parser = argparse.ArgumentParser(description='FDA baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=36)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--beta', type=float, default=0.01,
                        help='low-frequency swap ratio (official FDA default 0.01)')
    parser.add_argument('--input_size', type=int, default=224)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

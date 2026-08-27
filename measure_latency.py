# -*- coding: utf-8 -*-
"""Inference-time measurement for all comparison methods (paper Sec. 4.10).

Times each method over the whole target test set on three tasks (A->B / A->C / B->C);
reports total seconds, ms/image, params and FLOPs (Conv/Linear MACs x2).
Inference time depends only on the network structure, so trained weights are not loaded.
Discriminators/alignment modules exist only in training and are excluded (standard practice).

Usage (server):
    python measure_latency.py --methods DANN CDAN --repeats 3 --batch 16
Output: measure_latency_results.csv
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn

from models.fea_net import FEANet, Classifier
from comparison_experiments.common.models import (build_models, VGGBnBackbone,
                                                  VGGBnClassifier)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "measure_latency_results.csv")
INPUT_SIZE = 224
NUM_CLASSES = 3
TASKS = ["A->B", "A->C", "B->C"]
TGT_OF_TASK = {"A->B": "B", "A->C": "C", "B->C": "C"}


# ---- model builders ----
class FccanInference(nn.Module):
    """FCCAN: FEA-Net backbone (same as the main results) + 3-layer classifier head."""
    def __init__(self):
        super().__init__()
        self.enc = FEANet(wrb_alpha=0.4, wrb_lambda=0.3, use_hf_comp=True,
                          hf_comp_scale=0.2, use_msw_sa=True,
                          msw_sa_positions='3', use_wrb_after_layer2=True)
        self.clf = Classifier(self.enc.combined_features, NUM_CLASSES, prob=0.3)

    def forward(self, x):
        f, _ = self.enc(x)       # FEANet returns (features, features)
        logits, _ = self.clf(f)  # Classifier returns (logits, mid_out)
        return logits


def build_fccan():
    return FccanInference()


def build_thuml_resnet50():
    enc, clf = build_models(NUM_CLASSES, torch.device('cpu'), backbone='resnet50')
    return nn.Sequential(enc, clf)


def build_thuml_resnet50_dropout():
    enc, clf = build_models(NUM_CLASSES, torch.device('cpu'), backbone='resnet50',
                            bottleneck_style='dropout')
    return nn.Sequential(enc, clf)


def build_thuml_resnet18():
    enc, clf = build_models(NUM_CLASSES, torch.device('cpu'), backbone='resnet18')
    return nn.Sequential(enc, clf)


def build_thuml_vgg16():
    enc, clf = build_models(NUM_CLASSES, torch.device('cpu'), backbone='vgg16')
    return nn.Sequential(enc, clf)


def build_vgg16bn():
    return nn.Sequential(VGGBnBackbone(), VGGBnClassifier(NUM_CLASSES))


def build_dagcn():
    from comparison_experiments.dagcn.model import (DAGCNModel,
                                                    Classifier as DAGCNClassifier)
    enc = DAGCNModel()
    clf = DAGCNClassifier(enc.combined_features, NUM_CLASSES, prob=0.3)

    class W(nn.Module):
        def __init__(self, enc, clf):
            super().__init__()
            self.enc = enc
            self.clf = clf

        def forward(self, x):
            f, _ = self.enc(x)
            logits, _ = self.clf(f)
            return logits
    return W(enc, clf)


# Method name -> builder (Table 1 order, FCCAN last)
METHODS = [
    ("ResNet-18", build_thuml_resnet18),
    ("ResNet-50", build_thuml_resnet50),
    ("VGG-16", build_thuml_vgg16),
    ("DANN", build_thuml_resnet50),
    ("ADDA", build_vgg16bn),
    ("CDAN", build_thuml_resnet50),
    ("EM-DDA", build_vgg16bn),
    ("MCC", build_thuml_resnet50),
    ("SHOT", build_thuml_resnet50),
    ("CAT", build_vgg16bn),
    ("SVDNA", build_thuml_resnet50),
    ("DAGCN", build_dagcn),
    ("FCCAN (Ours)", build_fccan),
]


# ---- whole-test-set timing ----
def measure_testset_time(model, loader, repeats=3, warmup_batches=2):
    """Time inference over the whole test set (GPU); return (total_s, n_images).
    2 warmup batches, repeats averaged; timing covers the forward pass only."""
    model.eval()
    n_images = len(loader.dataset)
    acc_ms = 0.0
    with torch.no_grad():
        for _ in range(repeats):
            for i, (xs, _) in enumerate(loader):
                if i >= warmup_batches:
                    break
                if torch.cuda.is_available():
                    xs = xs.cuda()
                model(xs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for xs, _ in loader:
                if torch.cuda.is_available():
                    xs = xs.cuda()
                model(xs)
            torch.cuda.synchronize()
            acc_ms += (time.perf_counter() - t0) * 1e3
    return acc_ms / repeats / 1e3, n_images


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model, size=INPUT_SIZE):
    """Count MACs of Conv2d/Linear only; x2 -> FLOPs."""
    hooks = []
    macs = [0]

    def _hook_conv(module, inp, out):
        x = inp[0]
        k_h, k_w = module.kernel_size
        macs[0] += out.numel() * x.size(1) * k_h * k_w // module.groups

    def _hook_linear(module, inp, out):
        macs[0] += module.in_features * out.size(-1)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_hook_conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_hook_linear))
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.randn(1, 3, size, size, device=dev))
    for h in hooks:
        h.remove()
    return macs[0] * 2 / 1e9   # MACs -> FLOPs (G)



def main():
    ap = argparse.ArgumentParser(description='Inference-time measurement of all comparison methods on three tasks')
    ap.add_argument('--methods', nargs='+', default=None,
                    help='only measure the given methods (default: all)')
    ap.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS,
                    help='task list (default: all three)')
    ap.add_argument('--repeats', type=int, default=3,
                    help='repeats over the whole test set')
    ap.add_argument('--batch', type=int, default=16,
                    help='test inference batch size')
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dev = torch.device('cuda')
        print("GPU: {}  torch={}".format(torch.cuda.get_device_name(0), torch.__version__))
    else:
        dev = torch.device('cpu')
        print("[warn] no CUDA detected. torch={}".format(torch.__version__))
    print("input: {}x{}x3  batch={}  repeats={}  tasks={}".format(
        INPUT_SIZE, INPUT_SIZE, args.batch, args.repeats, args.tasks))
    print("-" * 100)

    from comparison_experiments.common.data_loader import load_task

    # Load the target test set once per task (shuffle=False, batched inference)
    test_loaders = {}
    for task in args.tasks:
        data = load_task('A', TGT_OF_TASK[task], input_size=INPUT_SIZE,
                         batch_src=args.batch, batch_tgt=args.batch,
                         tmi_target_unlabeled_pct=50)
        test_loaders[task] = data['tgt_test']
        print("task {} test set size: {}".format(task, len(data['tgt_test'].dataset)))

    methods = METHODS
    if args.methods:
        methods = [m for m in METHODS if m[0] in args.methods]

    rows = []
    for name, builder in methods:
        if name == "DAGCN":
            try:
                import torch_geometric  # noqa: F401
            except ImportError:
                print("[warn] torch_geometric not installed, skip DAGCN")
                continue
        per_task = {}
        for task in args.tasks:
            torch.manual_seed(42)
            model = builder().to(dev)
            params = count_params(model)
            try:
                flops = count_flops(model)
            except Exception:
                flops = float('nan')
            model = model.to(dev)
            total_s, n_img = measure_testset_time(model, test_loaders[task],
                                                  repeats=args.repeats)
            ms_per = total_s * 1e3 / n_img
            row = {"method": name, "task": task, "n_images": n_img,
                   "params_M": round(params, 2),
                   "flops_G": round(flops, 3) if flops == flops else 'nan',
                   "total_s": round(total_s, 3), "ms_per_image": round(ms_per, 3)}
            rows.append(row)
            per_task[task] = (total_s, ms_per)
            print("{:<16s} {:>4s}  n={}  total={:.3f}s  ms/img={:.3f}ms"
                  .format(name, task, n_img, total_s, ms_per))
        ts = [per_task[t][0] for t in args.tasks]
        m = sum(ts) / len(ts)
        sd = (sum((x - m) ** 2 for x in ts) / len(ts)) ** 0.5
        print("  >> {} total over 3 tasks = {:.3f}±{:.3f} s".format(name, m, sd))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "task", "n_images", "params_M",
                                          "flops_G", "total_s", "ms_per_image"])
        w.writeheader()
        w.writerows(rows)
    print("-" * 100)
    print("written to {}".format(OUT_CSV))


if __name__ == "__main__":
    main()

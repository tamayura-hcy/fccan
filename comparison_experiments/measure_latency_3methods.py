# -*- coding: utf-8 -*-
"""measure_latency_3methods.py —— TVT / DaC 目标域测试集推理耗时。

与表 V 同口径：batch 16、3 遍取均值、输出整个测试集总时间（s）。
模型结构与官方一致（权重值不影响推理耗时，采用随机初始化）。

用法：
    python -m comparison_experiments.measure_latency_3methods --method tvt --task A-B
    python -m comparison_experiments.measure_latency_3methods --method all
输出：
    comparison_experiments/results/latency_3methods.csv
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD = [0.229, 0.224, 0.225]

# 目标域测试集（与主表一致）
TASKS = {"A-B": "TMI", "A-C": "CELL", "B-C": "CELL"}
DATA_DIRS = {"TMI": "TMIdata_split_by_person", "CELL": "CELL_split_2025"}


def get_loader(task, batch=16, tf=None):
    tgt = TASKS[task]
    path = os.path.join(ROOT, "datasets", DATA_DIRS[tgt], "test")
    if tf is None:
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IM_MEAN, std=IM_STD)])
    ds = datasets.ImageFolder(path, transform=tf)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)


def build_tvt(task=None):
    sys.path.insert(0, os.path.join(ROOT, "comparison_experiments",
                                    "third_party", "TVT"))
    from models.modeling import VisionTransformer, CONFIGS, AdversarialNetwork
    cfg = CONFIGS["ViT-B_16"]
    model = VisionTransformer(cfg, 224, zero_head=True, num_classes=3, msa_layer=12)
    ad_net = AdversarialNetwork(cfg.hidden_size // 12, cfg.hidden_size // 12)
    # TVT 测试 transform（官方 office 分支：Resize(224) + 仅减均值）
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=IM_MEAN, std=[1.0, 1.0, 1.0])])
    return model, ad_net, tf


def build_dac(task=None):
    sys.path.insert(0, os.path.join(ROOT, "comparison_experiments",
                                    "third_party", "DaC", "VisDA"))
    import network
    netF = network.ResBase(res_name="resnet50")
    netB = network.feat_bootleneck(type="bn", feature_dim=netF.in_features,
                                   bottleneck_dim=256)
    netC = network.feat_classifier(type="wn", class_num=3, bottleneck_dim=256)
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IM_MEAN, std=IM_STD)])
    return (netF, netB, netC), None, tf


def forward(method, model, x, ad_net=None):
    if method == "tvt":
        logits, _, _ = model(x, ad_net=ad_net)
    elif method == "dac":
        netF, netB, netC = model
        logits = netC(netB(netF(x)))
    return logits


def measure(method, task, device="cuda", batch=16, repeats=3):
    model, ad_net, tf = {"tvt": build_tvt, "dac": build_dac}[method](task)
    if method == "dac":
        for m in model:
            m.to(device).eval()
    else:
        model.to(device).eval()
    if ad_net is not None:
        ad_net.to(device).eval()
    loader = get_loader(task, batch=batch, tf=tf)

    # 预热（首个 batch 含 CUDA 初始化）
    with torch.no_grad():
        for x, _ in loader:
            forward(method, model, x.to(device), ad_net)
            break

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for x, _ in loader:
                forward(method, model, x.to(device), ad_net)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all", choices=["tvt", "dac", "all"])
    ap.add_argument("--tasks", nargs="+", default=["A-B", "A-C", "B-C"])
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    methods = ["tvt", "dac"] if args.method == "all" else [args.method]
    out_csv = os.path.join(HERE, "results", "latency_3methods.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    for method in methods:
        for task in args.tasks:
            best, avg = measure(method, task, batch=args.batch)
            print("{} {}: best={:.3f}s avg={:.3f}s".format(method, task, best, avg))
            rows.append({"method": method, "task": task, "best_s": round(best, 4),
                         "avg_s": round(avg, 4)})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "task", "best_s", "avg_s"])
        w.writeheader()
        w.writerows(rows)
    print("已写入 {}".format(out_csv))


if __name__ == "__main__":
    main()

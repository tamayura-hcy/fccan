#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best config x 5 seeds batch (15 runs): full metrics + t-SNE + per-epoch diagnostics.

Output under best/: {task}_s{seed}/tsne.png, per_epoch.txt, metrics.txt,
and SUMMARY.txt (3-task mean±std).

Task configs (best.md, 2026-08-10): AB es=5 et=8; AC es=4 et=15; BC es=8 et=15.
Usage: python run_best_metrics.py [--seeds 42,123,777,2024,3407] [--log run.log]
"""
import argparse
import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
OUT = os.path.join(ROOT, "best")
CELL_SPLIT = "CELL_split_2025"
SEEDS = [42, 123, 777, 2024, 3407]

BEST_TASKS = [
    {"tag": "A-B", "only": "BOE->TMI", "es": 5, "et": 8,
     "extra": ["--src_ll_prob", "0.7", "--src_ll_alpha", "2.0"]},
    {"tag": "A-C", "only": "BOE->CELL", "es": 4, "et": 15,
     "extra": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.001", "--alpha_scon", "0.01"]},
    {"tag": "B-C", "only": "TMI->CELL", "es": 8, "et": 15,
     "extra": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.5"]},
]

METRIC_KEYS = ['acc', 'auc', 'recall', 'precision', 'f1', 'bacc',
               'specificity', 'kappa', 'gmean', 'mcc']
TEST_RE = re.compile(
    r'test acc=(-?[\d.]+) \| auc=(-?[\d.]+) \| recall=(-?[\d.]+) \| precision=(-?[\d.]+) \| f1=(-?[\d.]+)'
    r' \| bacc=(-?[\d.]+) \| specificity=(-?[\d.]+) \| kappa=(-?[\d.]+) \| gmean=(-?[\d.]+) \| mcc=(-?[\d.]+)')


def _copy_artifacts(tag, seed, src_label, tgt_label, out_dir, log_handle):
    """Copy t-SNE and per-epoch diagnostics from main.py to best/{tag}_s{seed}/."""
    # 1) t-SNE：saves/FEANet_{src}_to_{tgt}_iter1/tsne.png
    save_name = os.path.join("saves", "FEANet_{}_to_{}_iter1".format(src_label, tgt_label))
    tsne_src = os.path.join(save_name, "tsne.png")
    if os.path.exists(tsne_src):
        shutil.copy2(tsne_src, os.path.join(out_dir, "tsne.png"))
        print("  [t-SNE] saved -> {}/tsne.png".format(out_dir))
    else:
        print("  [WARN] t-SNE not found: {}".format(tsne_src))

    # 2) per-epoch diagnostics txt under results/result_logs/*/{src}_to_{tgt}/iter1_seed{seed}*.txt
    txts = glob.glob(os.path.join(
        "results", "result_logs", "*", "*{}_to_{}*".format(src_label, tgt_label),
        "iter1_seed{}*.txt".format(seed)))
    if txts:
        shutil.copy2(txts[0], os.path.join(out_dir, "per_epoch.txt"))
        print("  [per-epoch] saved -> {}/per_epoch.txt".format(out_dir))
    else:
        print("  [WARN] per-epoch txt not found (--save_result_txt off?)")


def run_one(tag, cfg, seed, log_handle):
    only = cfg["only"]
    src_label, tgt_label = only.split("->")
    out_dir = os.path.join(OUT, "{}_s{}".format(tag, seed))
    os.makedirs(out_dir, exist_ok=True)

    cmd = [sys.executable, "main.py", "--only", only,
           "-es", str(cfg["es"]), "-et", str(cfg["et"]),
           "--seed", str(seed), "-i", "1",
           "--cell_split", CELL_SPLIT,
           "--save_result_txt", "1"] + cfg["extra"]
    print("\n[RUN] {} s{}  only={} es={} et={}".format(
        tag, seed, only, cfg["es"], cfg["et"]))
    print("    cmd: python {} {}".format(cmd[0], " ".join(cmd[1:])))
    if log_handle:
        log_handle.write("\n[RUN] {} s{}  only={} es={} et={}\n".format(
            tag, seed, only, cfg["es"], cfg["et"]))
        log_handle.flush()

    t0 = time.time()
    last_m = None
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.rstrip()
            if ('%|' in line) or ('it/s' in line):
                continue            # filter tqdm progress bars
            if line:
                print(line)
            if log_handle:
                log_handle.write(line + "\n")
            _m = TEST_RE.search(line)
            if _m:
                last_m = {k: float(v) for k, v in zip(METRIC_KEYS, _m.groups())}
        proc.wait()
        elapsed = time.time() - t0
    except Exception as e:
        print("[ERROR] {} s{}: {}".format(tag, seed, e))
        if log_handle:
            log_handle.write("[ERROR] {} s{}: {}\n".format(tag, seed, e))
            log_handle.flush()
        return None
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()

    if proc.returncode != 0:
        print("[FAIL] {} s{} exit={}".format(tag, seed, proc.returncode))
        return None

    # Write final metrics to metrics.txt
    if last_m:
        with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write("task={} seed={}  final metrics ({})\n".format(
                tag, seed, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            for k in METRIC_KEYS:
                f.write("  {:<12} {:.4f}\n".format(k, last_m[k]))
    _copy_artifacts(tag, seed, src_label, tgt_label, out_dir, log_handle)

    msg = "[OK] {} s{} done ({}s)".format(tag, seed, int(elapsed))
    print(msg)
    if log_handle:
        log_handle.write(msg + "\n")
        log_handle.flush()
    return last_m


def main():
    ap = argparse.ArgumentParser(description="Best config x 5 seeds batch (output under best/)")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)),
                    help="seed list (comma-separated)")
    ap.add_argument("--log", type=str, default=None, help="log file (also written to output)")
    ap.add_argument("--tasks", type=str, default=None,
                    help="only run these tasks (comma-separated, e.g. A-B,B-C)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = [t for t in BEST_TASKS] if not args.tasks else \
        [t for t in BEST_TASKS if t["tag"] in [x.strip() for x in args.tasks.split(",")]]

    os.makedirs(OUT, exist_ok=True)
    log_handle = open(args.log, "a", encoding="utf-8") if args.log else None

    print("best-config batch: {} tasks x {} seeds = {} runs".format(
        len(tasks), len(seeds), len(tasks) * len(seeds)))
    print("output dir: {}".format(OUT))
    if log_handle:
        log_handle.write("# best batch start {} tasks={} seeds={}\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), [t["tag"] for t in tasks], seeds))
        log_handle.flush()

    results = {t["tag"]: {} for t in tasks}
    for cfg in tasks:
        for seed in seeds:
            m = run_one(cfg["tag"], cfg, seed, log_handle)
            if m is not None:
                results[cfg["tag"]][seed] = m

    # ---- SUMMARY.txt ----
    summary_path = os.path.join(OUT, "SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# FEA-Net best-config full-metrics summary (best)\n")
        f.write("# generated: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("# task configs:\n")
        for cfg in tasks:
            f.write("#   {:<4} {}  es={} et={}  extra={}\n".format(
                cfg["tag"], cfg["only"], cfg["es"], cfg["et"], " ".join(cfg["extra"])))
        f.write("\n")
        for cfg in tasks:
            tag = cfg["tag"]
            d = results[tag]
            f.write("\n### {}\n".format(tag))
            f.write("  | seed | " + " | ".join(METRIC_KEYS) + " |\n")
            f.write("  |---|" + "---|" * len(METRIC_KEYS) + "\n")
            for seed in seeds:
                if seed in d:
                    m = d[seed]
                    f.write("  | {} | {} |\n".format(
                        seed, " | ".join("{:.4f}".format(m[k]) for k in METRIC_KEYS)))
            vals = {k: [d[s][k] for s in seeds if s in d] for k in METRIC_KEYS}
            if any(vals.values()):
                n = max(len(v) for v in vals.values())
                f.write("  | **mean** | {} |\n".format(
                    " | ".join("{:.4f}".format(statistics.mean(vals[k]))
                               if vals[k] else "—" for k in METRIC_KEYS)))
                f.write("  | **std**  | {} |\n".format(
                    " | ".join("{:.4f}".format(statistics.stdev(vals[k]))
                               if len(vals[k]) > 1 else "—" for k in METRIC_KEYS)))
    print("\nsummary written: {}".format(summary_path))

    if log_handle:
        log_handle.close()
    print("\n===== all done, output in {} =====".format(OUT))


if __name__ == "__main__":
    main()

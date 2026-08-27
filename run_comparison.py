#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified comparison runner: 11 methods (10 baselines + FEA-Net) x 3 tasks x 5 seeds.

Tasks: A-B BOE->TMI, A-C BOE->CELL, B-C TMI->CELL. Seeds: 42,123,777,2024,3407.
Baselines: dann/adda/cdan/mcc/shot/svdna/emdda/dagcn/cat + 3 source-only runs,
sharing the ResNet50 backbone, data split and evaluation under common/.
FEA-Net: python main.py --only "BOE->TMI" -es 5 -et 30 --seed S -i 1.

Results: results_comparison_all.csv
Usage:
    python run_comparison.py --tasks A-C / --methods dann,cdan / --rebuild
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 123, 777, 2024, 3407]
# All three tasks are re-run with the updated comparison code to keep a consistent protocol.
TASKS = ["A-B", "A-C", "B-C"]
_CELL_SPLIT = "CELL_split_2025"     # set by --cell_split, passed to subprocesses via env
TASK = "A-B"                        # current task (set by the --tasks loop)
SRC, TGT = TASK.split("-")
TASK_ONLY = {"A-B": "BOE->TMI", "A-C": "BOE->CELL", "B-C": "TMI->CELL"}
COMPARE_METHODS = ["dann", "adda", "cdan", "mcc", "shot", "svdna", "emdda", "dagcn",
                   "cat",
                   "src_only_resnet50", "src_only_vgg16", "src_only_resnet18"]
# Default set: exclude adda/emdda (implementation issues), run the rest on all three tasks.
DEFAULT_METHODS = [m for m in COMPARE_METHODS if m not in ("adda", "emdda")]
# Source-only baselines go through comparison_experiments.source_only.train.
SRC_ONLY_BACKBONE = {
    "src_only_resnet50": "resnet50",
    "src_only_vgg16": "vgg16",
    "src_only_resnet18": "resnet18",
}
EPOCHS = 10            # fallback for methods not in METHOD_CFG
BATCH = 16
# Official per-method configs (verified against official sources, 2026-08-08):
# thuml/TLL (DANN/CDAN/DAN/MCC) 20ep bs32 (MCC 36); cuishuhao/BNM 20ep bs36;
# tim-learn/SHOT src20/tgt15 bs64; xuqing88/OCT-DDA (ADDA/EM-DDA) 30ep bs8;
# SVDNA self-implemented (10ep/16). Batch sizes assume resnet18; we use resnet50,
# override with --batch if OOM, keep epochs at official values.
METHOD_CFG = {
    'dann':  {'epochs': 20, 'batch': 32},
    'cdan':  {'epochs': 20, 'batch': 32},
    'mcc':   {'epochs': 20, 'batch': 36},
    'shot':  {'epochs': 15, 'batch': 64, 'src_epochs': 20},
    'svdna': {'epochs': 10, 'batch': 16},
    'adda':  {'epochs': 10, 'batch': 8, 'src_epochs': 15},
    'emdda': {'epochs': 10, 'batch': 8, 'src_epochs': 15},
    'dagcn': {'epochs': 10, 'batch': 32, 'src_epochs': 15},
    # CAT: OCT-DDA official (Deng 2019 ICCV); 2026-08-09: epochs 30->15 early stop.
    'cat':   {'epochs': 15, 'batch': 8},
}
# DAGCN loss coefficients and classifier lr are fixed inside train.py (Eq. 8).
# FEA-Net config (matches the 30-seed experiments); per-task overrides:
#   A-B: default; A-C: user-tuned; B-C: early stop at 15 epochs.
FEA_ES = 5
FEA_ET = 30
FEA_OVERRIDES = {
    # A-B: default + low-frequency augmentation prob 0.8
    "A-B": ["--src_ll_prob", "0.8"],
    "A-C": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.001",
            "-et", "15", "--alpha_scon", "0.01"],   # A-C et=15 per best.md
    "B-C": ["-et", "15"],
}
RESULTS_CSV = os.path.join(ROOT, "results_comparison_all.csv")

METRIC_KEYS = ["acc", "auc", "recall", "precision", "f1", "bacc",
               "specificity", "kappa", "gmean", "mcc"]
# Single CSV: type=summary for baselines, type=epoch for FEA-Net per-epoch STUDENT/EMA.
CSV_HEADER = ["type", "method", "task", "seed", "epoch", "model"] + METRIC_KEYS
TEST_METRICS_RE = re.compile(
    r"test acc=(-?[\d.]+) \| auc=(-?[\d.]+) \| recall=(-?[\d.]+) \| precision=(-?[\d.]+) \| "
    r"f1=(-?[\d.]+) \| bacc=(-?[\d.]+) \| specificity=(-?[\d.]+) \| kappa=(-?[\d.]+) \| "
    r"gmean=(-?[\d.]+) \| mcc=(-?[\d.]+)")


def _ensure_csv_header():
    """Write the CSV header when missing or empty."""
    if not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)


def _append_rows(rows):
    """Append rows to the CSV."""
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def _dedup_csv():
    """Dedup by (type, method, task, seed, epoch, model), keeping the last row."""
    if not os.path.exists(RESULTS_CSV):
        return
    with open(RESULTS_CSV, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f)]
    if not rows:
        return
    header, data = rows[0], rows[1:]
    seen = {}
    for r in data:
        if len(r) < 4:
            continue
        # key = all id columns (type, method, task, seed, epoch, model)
        key = (r[0], r[1], r[2], r[3], r[4] if len(r) > 4 else "",
               r[5] if len(r) > 5 else "")
        seen[key] = r
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in seen.values():
            w.writerow(r)
    print("  [CSV] dedup done: {} rows -> {} rows".format(len(data), len(seen)))


def _parse_full_metrics(out):
    """Parse all full-metric lines from output; return (list, last dict)."""
    metrics = []
    for line in out.splitlines():
        m = TEST_METRICS_RE.search(line)
        if m:
            metrics.append({k: float(v) for k, v in zip(METRIC_KEYS, m.groups())})
    return metrics, (metrics[-1] if metrics else None)


def run_compare(method, seed, skip_existing):
    out_dir = os.path.join(ROOT, "comparison_experiments", "results",
                           "{}__{}__s{}".format(method, TASK, seed))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    if skip_existing and os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            _, last = _parse_full_metrics(f.read())
            return last
    _mod = method
    _extra = []
    if method in SRC_ONLY_BACKBONE:
        _mod = "source_only"
        _extra = ["--backbone", SRC_ONLY_BACKBONE[method]]
    # Official per-method config; explicit --epochs/--batch override it.
    _cfg = METHOD_CFG.get(method, {})
    _ep = str(_cfg.get('epochs', EPOCHS))
    _bs = str(_cfg.get('batch', BATCH))
    cmd = [sys.executable, "-m",
           "comparison_experiments.{}.train".format(_mod),
           "--src", SRC, "--tgt", TGT, "--seed", str(seed),
           "--epochs", _ep, "--batch", _bs] + _extra
    if method == "dagcn":
        # DAGCN official: src 15ep, bs 32 per domain, input 224; B->C target 15ep.
        _tgt_ep = "15" if TASK == "B-C" else _ep
        cmd += ["--src_epochs", str(_cfg.get('src_epochs', 15)), "--input_size", "224",
                "--epochs", _tgt_ep]
    elif method == "shot":
        # SHOT: src 20ep / tgt 15ep.
        cmd += ["--src_epochs", str(_cfg.get('src_epochs', 20))]
    elif method in ("adda", "emdda"):
        # Early stop per user decision (2026-08-14): src 15ep + tgt 10ep with best-epoch saving.
        cmd += ["--src_epochs", str(_cfg.get('src_epochs', 15))]
    print("RUN {} {} s{}".format(method, TASK, seed))
    _run_tee(cmd, log_path)
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        _, last = _parse_full_metrics(f.read())
    if last is None:
        _print_log_tail(log_path, method, seed)
    return last


def _run_tee(cmd, log_path):
    """Run cmd, tee output to terminal and log file. PYTHONUNBUFFERED=1 for live output."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CELL_SPLIT"] = _CELL_SPLIT
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env, encoding="utf-8", errors="replace")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
            print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        print("  [WARN] {} exit code {}\n".format(os.path.basename(cmd[2]), proc.returncode))


def _print_log_tail(log_path, method, seed):
    """Print log tail when metrics could not be parsed."""
    print("  [WARN] {} {} s{} no metrics parsed, log tail:".format(method, TASK, seed))
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            tail = f.readlines()[-15:]
        for ln in tail:
            print("    " + ln.rstrip())
    except Exception:
        pass


def run_feanet(seed, skip_existing):
    out_dir = os.path.join(ROOT, "comparison_experiments", "results",
                           "feanet__{}__s{}".format(TASK, seed))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    if skip_existing and os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            return _parse_feanet(f.read())
    cmd = [sys.executable, "main.py", "--only", TASK_ONLY[TASK],
           "-es", str(FEA_ES), "-et", str(FEA_ET),
           "--seed", str(seed), "-i", "1", "--cell_split", _CELL_SPLIT,
           "--save_which", "ema"]   # EMA teacher is the main metric (eval/save EMA weights)
    cmd += FEA_OVERRIDES.get(TASK, [])   # per-task FEA-Net overrides (later argparse wins)
    print("RUN FEA-Net {} s{}".format(TASK, seed))
    _run_tee(cmd, log_path)
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    res = _parse_feanet(text)
    if res[0] is None and res[2] is None:
        _print_log_tail(log_path, "FEA-Net", seed)
    return res


def _parse_feanet_epochs(text):
    """Parse per-epoch STUDENT/EMA full metrics; return {epoch: {'student': ..., 'ema': ...}}."""
    last_m = None
    epochs = {}
    for line in text.splitlines():
        m = TEST_METRICS_RE.search(line)
        if m:
            last_m = {k: float(v) for k, v in zip(METRIC_KEYS, m.groups())}
        if "student_acc=" in line:
            mep = re.search(r"epoch\s*(\d+)/", line)
            ep = int(mep.group(1)) if mep else (len(epochs) + 1)
            mval = re.search(r"student_acc=([\d.]+)", line)
            val = float(mval.group(1)) if mval else None
            rec = epochs.setdefault(ep, {"student": None, "ema": None})
            rec["student"] = dict(last_m) if last_m else {}
            if val is not None:
                rec["student"]["acc"] = val
        if "ema_acc=" in line:
            mep = re.search(r"epoch\s*(\d+)/", line)
            ep = int(mep.group(1)) if mep else (len(epochs) + 1)
            mval = re.search(r"ema_acc=([\d.]+)", line)
            val = float(mval.group(1)) if mval else None
            rec = epochs.setdefault(ep, {"student": None, "ema": None})
            rec["ema"] = dict(last_m) if last_m else {}
            if val is not None:
                rec["ema"]["acc"] = val
    return epochs


def _write_feanet_epochs_csv():
    """Scan feanet result dirs and rebuild results_feanet_epochs.csv (legacy)."""
def _write_feanet_epochs_csv():
    """Scan feanet result dirs, append per-epoch STUDENT/EMA rows to the unified CSV."""
    results_dir = os.path.join(ROOT, "comparison_experiments", "results")
    rows = []
    if os.path.isdir(results_dir):
        for d in sorted(os.listdir(results_dir)):
            m = re.match(r"feanet__(A-B|A-C|B-C)__s(\d+)$", d)
            if not m:
                continue
            task, seed = m.group(1), int(m.group(2))
            log_path = os.path.join(results_dir, d, "run.log")
            if not os.path.exists(log_path):
                continue
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                epochs = _parse_feanet_epochs(f.read())
            for ep in sorted(epochs):
                for model in ["student", "ema"]:
                    rec = epochs[ep][model]
                    if rec:
                        rows.append(["epoch", "FEA-Net", task, seed, ep, model] +
                                    [rec.get(k, "") for k in METRIC_KEYS])
    _append_rows(rows)
    print("  [FEA-Net] per-epoch rows appended to CSV ({} rows)".format(len(rows)))


def _parse_feanet(text):
    """Parse main.py log: Student BEST acc, EMA last-epoch metrics, SWA metrics."""
    stu = []
    ema_final_metrics = None
    metrics_list = []
    swa_candidates = []
    for line in text.splitlines():
        m = TEST_METRICS_RE.search(line)
        if m:
            metrics_list.append({k: float(v) for k, v in zip(METRIC_KEYS, m.groups())})
        if "student_acc=" in line:
            mm = re.search(r"student_acc=([\d.]+)", line)
            if mm:
                stu.append(float(mm.group(1)))
        if "ema_acc=" in line:
            mm = re.search(r"ema_acc=([\d.]+)", line)
            if mm:
                if metrics_list:
                    ema_final_metrics = dict(metrics_list[-1])
                    ema_final_metrics["acc"] = float(mm.group(1))
                else:
                    ema_final_metrics = {k: None for k in METRIC_KEYS}
                    ema_final_metrics["acc"] = float(mm.group(1))
        if "swa_acc=" in line:
            if metrics_list:
                swa_candidates.append(metrics_list[-1])
    best_stu = max(stu) if stu else None
    swa_metrics = swa_candidates[-1] if swa_candidates else (metrics_list[-1] if metrics_list else None)
    return best_stu, ema_final_metrics, swa_metrics


def run_one_task(task, methods, seeds, skip_existing, with_feanet=False):
    """Run one task: methods x seeds; append everything to the unified CSV."""
    global TASK, SRC, TGT
    TASK = task
    SRC, TGT = task.split("-")
    print("\n" + "=" * 90)
    print("[Task] {} = {} -> {}   cell_split={}   start".format(TASK, SRC, TGT, _CELL_SPLIT))
    print("=" * 90)

    rows = []   # [method, task, seed, metrics_dict] for console summary

    def _write_summary(method, task, seed, md):
        _append_rows([["summary", method, task, seed, "", ""] +
                      ([md.get(k) for k in METRIC_KEYS] if md else [None] * len(METRIC_KEYS))])

    # 1) Baselines (full metrics dict)
    for method in methods:
        for seed in seeds:
            md = run_compare(method, seed, skip_existing)
            rows.append([method, TASK, seed, md])
            _write_summary(method, TASK, seed, md)
    # 2) FEA-Net (only with --with-feanet): Student BEST / EMA last / SWA
    if with_feanet:
        for seed in seeds:
            best_stu, ema_final, swa_metrics = run_feanet(seed, skip_existing)
            stu_dict = None
            if best_stu is not None:
                stu_dict = {k: (best_stu if k == "acc" else None) for k in METRIC_KEYS}
            rows.append(["FEA-Net (Student BEST)", TASK, seed, stu_dict])
            rows.append(["FEA-Net (EMA)", TASK, seed, ema_final])
            rows.append(["FEA-Net (SWA)", TASK, seed, swa_metrics])
            _write_summary("FEA-Net (Student BEST)", TASK, seed, stu_dict)
            _write_summary("FEA-Net (EMA)", TASK, seed, ema_final)
            _write_summary("FEA-Net (SWA)", TASK, seed, swa_metrics)
        # Append per-epoch rows (type=epoch).
        _write_feanet_epochs_csv()

    print("saved -> {}".format(RESULTS_CSV))

    print("\n=== Summary {} (mean±std over {} seeds, full metrics) ===".format(TASK, len(seeds)))
    method_labels = methods
    if with_feanet:
        method_labels = method_labels + ["FEA-Net (Student BEST)", "FEA-Net (EMA)", "FEA-Net (SWA)"]
    for method in method_labels:
        line = "  {:24s} {}  ".format(method, TASK)
        n = None
        for k in METRIC_KEYS:
            vals = [r[3][k] for r in rows if r[0] == method and r[3] and r[3].get(k) is not None]
            if vals:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                n = len(vals)
                line += "{}={:.4f}±{:.4f} ".format(k, mean, std)
        if n:
            print(line + "n={}".format(n))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Comparison: 10 methods (deliverable subset) x tasks x 5 seeds")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS),
                        help="comma-separated method subset; default excludes adda/emdda")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)),
                        help="comma-separated seed subset; default 42,123,777,2024,3407")
    parser.add_argument("--tasks", type=str, default=",".join(TASKS),
                        help="comma-separated task subset; default A-B,A-C,B-C")
    parser.add_argument("--cell_split", type=str, default="CELL_split_2025",
                        choices=["CELL_split_2025", "CELL_split_502525"],
                        help="CELL dataset split (passed to subprocesses)")
    parser.add_argument("--with-feanet", action="store_true",
                        help="also run FEA-Net (off by default)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the CSV (clear old data); default appends")
    args = parser.parse_args()
    global _CELL_SPLIT
    _CELL_SPLIT = args.cell_split

    methods = [m for m in args.methods.split(",") if m in COMPARE_METHODS]
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = [t for t in args.tasks.split(",") if t in TASKS]

    # Results go to results_comparison_all.csv; append by default, --rebuild clears.
    if args.rebuild:
        if os.path.exists(RESULTS_CSV):
            os.remove(RESULTS_CSV)
    _ensure_csv_header()

    print("[Global] tasks={}  methods={}  seeds={}  cell_split={}  with_feanet={}".format(
        tasks, methods, seeds, _CELL_SPLIT, args.with_feanet))
    t_all = time.time()
    for task in tasks:
        # No cached logs locally, so run everything (skip_existing=False).
        run_one_task(task, methods, seeds, skip_existing=False, with_feanet=args.with_feanet)
        print("\n[Task] {} done (cumulative {:.0f}s)".format(task, time.time() - t_all))
    print("\nAll tasks done: {} (total {:.1f} min)".format(tasks, (time.time() - t_all) / 60))
    # Dedup: overwritten methods keep only the latest row per (method, task, seed).
    _dedup_csv()


if __name__ == "__main__":
    import time
    main()

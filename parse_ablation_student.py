# -*- coding: utf-8 -*-
"""Parse the no_ema Student final-epoch metrics from ablation logs (for M2).

no_ema (--ema_guide_caco 0) disables EMA creation, so logs have no ema_acc=;
the final eval falls back to the Student. This script takes the last target-test
full-metric line from existing run.log files as the Student final epoch.

Usage:
    python parse_ablation_student.py --task AC,BC --logdir logs_ablation_3tasks --write
"""
import os, re, sys, glob, statistics, argparse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(HERE, "results_ablation_3tasks.txt")

TEST_METRICS_RE = re.compile(
    r'test acc=(-?[\d.]+) \| auc=(-?[\d.]+) \| recall=(-?[\d.]+) \| precision=(-?[\d.]+) \| f1=(-?[\d.]+)'
    r' \| bacc=(-?[\d.]+) \| specificity=(-?[\d.]+) \| kappa=(-?[\d.]+) \| gmean=(-?[\d.]+) \| mcc=(-?[\d.]+)')
METRIC_KEYS = ['acc', 'auc', 'recall', 'precision', 'f1', 'bacc',
               'specificity', 'kappa', 'gmean', 'mcc']

TASK_TAGS = {"AB": "BOE->TMI", "AC": "BOE->CELL", "BC": "TMI->CELL"}


def parse_student_final(log_path):
    """Parse the last target-test full metrics (Student final under no_ema)."""
    last_m = None
    in_target_test = False
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if "Test tgt_encoder + classifier on Target Test dataset" in line:
                    in_target_test = True
                    continue
                if in_target_test:
                    _m = TEST_METRICS_RE.search(line)
                    if _m:
                        last_m = {k: float(v) for k, v in zip(METRIC_KEYS, _m.groups())}
                        # keep in_target_test after a full-metric line (one eval can span lines),
                        # but exit at the next epoch marker
                    elif "epoch" in line and ("ema_acc" in line or "student_acc" in line or "==" in line):
                        in_target_test = False
    except Exception as e:
        print("[warn] read {}: {}".format(log_path, e))
    return last_m


def main():
    ap = argparse.ArgumentParser(description="Parse no_ema Student final epoch")
    ap.add_argument("--task", type=str, default=None, help="task subset AB/AC/BC")
    ap.add_argument("--logdir", type=str, default=os.path.join(HERE, "logs_ablation_3tasks"))
    ap.add_argument("--write", action="store_true", help="write to results_ablation_3tasks.txt")
    args = ap.parse_args()

    logdir = args.logdir
    tasks = list(TASK_TAGS) if not args.task else \
            [t.strip() for t in args.task.split(",") if t.strip() in TASK_TAGS]

    print("# parse no_ema Student final {} | logdir={}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), logdir))

    out_lines = ["\n=== no_ema Student final (log parse, for the M2 w/o-EMA ablation row) ==="]
    for tk in tasks:
        accs, aucs, logs_found = [], [], []
        for seed in [42, 123, 777, 2024, 3407]:
            g = glob.glob(os.path.join(logdir, "no_ema__{}__s{}".format(tk, seed), "run.log"))
            if not g:
                print("[missing] {} s{}".format(tk, seed))
                continue
            m = parse_student_final(g[0])
            if m is None:
                print("[no-data] {} s{}".format(tk, seed))
                continue
            accs.append(m['acc']); aucs.append(m['auc']); logs_found.append(seed)
            print("{:4s} s{:5d}  Student acc={:.4f} auc={:.4f}".format(tk, seed, m['acc'], m['auc']))
        if len(accs) == 5:
            acc_m, acc_s = statistics.mean(accs), statistics.stdev(accs)
            auc_m, auc_s = statistics.mean(aucs), statistics.stdev(aucs)
            line = ("no_ema(Student) {:4s}  acc={:.4f}±{:.4f}  auc={:.4f}±{:.4f}  seeds={}\n"
                    .format(tk, acc_m, acc_s, auc_m, auc_s, logs_found))
            print(line.strip())
            out_lines.append(line.strip())
        else:
            print("[incomplete] {} only {} seeds: {}".format(tk, len(accs), logs_found))
            out_lines.append("[incomplete] {} only {} seeds: {}".format(tk, len(accs), logs_found))

    if args.write and len(out_lines) > 1:
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print("\nwritten to {}".format(RESULTS_FILE))


if __name__ == "__main__":
    main()

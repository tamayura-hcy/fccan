# -*- coding: utf-8 -*-
"""Batch reproduction of TVT: 3 tasks x 5 seeds.

Protocol: TVT is single-stage (no source/target separation), official VisDA protocol
5000 steps, evaluate every 100 steps, record target-test acc to history CSV.

Usage:
    python -m comparison_experiments.tvt.run_all [--tasks A-B A-C B-C] [--num-steps 5000]
Output: comparison_experiments/tvt/results_tvt.csv and history_tvt.csv
"""
import argparse
import csv
import os
import sys

# Windows GBK console: use replacement chars for subprocess output to avoid UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from comparison_experiments.tvt.train import run_one, SEEDS
from comparison_experiments.tvt.make_lists import main as make_lists_main

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "results_tvt.csv")
HIST_CSV = os.path.join(HERE, "history_tvt.csv")
HISTORY_FIELDS = ["task", "seed", "round", "acc", "auc"]


def early_stop_summary(history_rows, task, seeds):
    """Mean acc of 5 seeds per round; return the best round's (best_round, acc_mean, acc_sd, auc_mean, auc_sd)."""
    from collections import defaultdict
    by_round = defaultdict(lambda: defaultdict(list))
    for r in history_rows:
        if r["task"] == task and r["seed"] in seeds:
            by_round[r["round"]]["acc"].append(r["acc"])
            if r.get("auc") is not None:
                by_round[r["round"]]["auc"].append(r["auc"])
    best_round, best_mean = None, -1.0
    for rnd in sorted(by_round):
        accs = by_round[rnd]["acc"]
        if len(accs) < len(seeds):
            continue
        m = sum(accs) / len(accs)
        if m > best_mean:
            best_mean, best_round = m, rnd
    if best_round is None:
        return None, None, None, None, None
    accs = by_round[best_round]["acc"]
    aucs = by_round[best_round]["auc"]
    m = sum(accs) / len(accs)
    sd = (sum((a - m) ** 2 for a in accs) / len(accs)) ** 0.5
    am = sum(aucs) / len(aucs) if aucs else float('nan')
    asd = (sum((a - am) ** 2 for a in aucs) / len(aucs)) ** 0.5 if aucs else float('nan')
    return best_round, m, sd, am, asd


def main():
    ap = argparse.ArgumentParser(description='TVT batch reproduction')
    ap.add_argument('--tasks', nargs='+', default=['A-B', 'A-C', 'B-C'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--num-steps', type=int, default=5000)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--batch', type=int, default=32, help='TVT training batch (32 for 16GB GPU, 16 for smaller GPU)')
    args = ap.parse_args()

    # regenerate lists to avoid stale absolute paths from another machine
    make_lists_main()

    rows = []
    history_rows = []
    seeds = [int(s) for s in args.seeds]
    for task in args.tasks:
        for seed in args.seeds:
            acc, auc, secs = run_one(task, seed, num_steps=args.num_steps,
                                     img_size=args.img_size, batch=args.batch,
                                     history_rows=history_rows)
            rows.append({"task": task, "seed": seed,
                         "best_acc": acc, "best_auc": auc, "seconds": secs})
        br, am, asd, aum, ausd = early_stop_summary(history_rows, task, seeds)
        if br is not None:
            print("== {} TVT early-stop: round={} acc={:.2f}±{:.2f}% auc={:.2f}±{:.2f}%".format(
                task, br, am * 100, asd * 100, aum * 100, ausd * 100))
        else:
            print("== {} TVT = no valid early-stop round".format(task))

    # summary CSV: official step-level best (for reference)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "best_acc", "best_auc", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print("Saved to {}".format(OUT_CSV))

    # early-stop CSV: used in the main table (one row per task)
    es_csv = os.path.join(HERE, "results_tvt_earlystop.csv")
    es_rows = []
    for task in args.tasks:
        br, am, asd, aum, ausd = early_stop_summary(history_rows, task, seeds)
        es_rows.append({"task": task, "best_round": br,
                        "acc_mean": round(am, 6) if am is not None else None,
                        "acc_sd": round(asd, 6) if asd is not None else None,
                        "auc_mean": round(aum, 6) if aum is not None else None,
                        "auc_sd": round(ausd, 6) if ausd is not None else None})
    with open(es_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "best_round", "acc_mean",
                                          "acc_sd", "auc_mean", "auc_sd"])
        w.writeheader()
        w.writerows(es_rows)
    print("Saved to {}".format(es_csv))

    with open(HIST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        w.writerows(history_rows)
    print("Saved to {} ({} rounds)".format(HIST_CSV, len(history_rows)))


if __name__ == "__main__":
    main()

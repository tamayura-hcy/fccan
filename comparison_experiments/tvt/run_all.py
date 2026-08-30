# -*- coding: utf-8 -*-
"""run_all.py —— TVT 三任务 × 5 种子批量复现。

协议：TVT 为单阶段方法（无源/目标分离），官方 VisDA 协议 5000 步，
每 100 步评估一轮，逐轮记录目标测试集 acc 到 history CSV。

用法：
    python -m comparison_experiments.tvt.run_all [--tasks A-B A-C B-C] [--num-steps 5000]
输出：
    comparison_experiments/tvt/results_tvt.csv    # 汇总（每任务每种子 acc + 均值±std）
    comparison_experiments/tvt/history_tvt.csv    # 每轮记录（round, acc）
"""
import argparse
import csv
import os
import sys

# 服务器 Windows 控制台 GBK：打印子进程输出含特殊字符时用替换符，避免 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from comparison_experiments.tvt.train import run_one, SEEDS
from comparison_experiments.tvt.make_lists import main as make_lists_main

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "results_tvt.csv")
HIST_CSV = os.path.join(HERE, "history_tvt.csv")
HISTORY_FIELDS = ["task", "seed", "round", "acc", "auc"]


def early_stop_summary(history_rows, task, seeds):
    """逐轮 5 种子 acc 均值，取均值最高轮，返回 (best_round, acc_mean, acc_sd, auc_mean, auc_sd)。"""
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
    ap = argparse.ArgumentParser(description='TVT 批量复现')
    ap.add_argument('--tasks', nargs='+', default=['A-B', 'A-C', 'B-C'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--num-steps', type=int, default=5000)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--batch', type=int, default=32, help='TVT 训练 batch（16GB 卡建议 32，更小卡用 16）')
    args = ap.parse_args()

    # 重新生成列表：避免同步过来的列表残留旧机器绝对路径
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
            print("== {} TVT = 无有效早停轮".format(task))

    # 汇总 CSV：官方步级 best（供参考）
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "best_acc", "best_auc", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print("已写入 {}".format(OUT_CSV))

    # 早停口径 CSV：主表用（每任务一行：最优轮 + acc/auc 均值±std）
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
    print("已写入 {}".format(es_csv))

    with open(HIST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        w.writerows(history_rows)
    print("已写入 {}（{} 轮记录）".format(HIST_CSV, len(history_rows)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""run_all.py —— DaC 三任务 × 5 种子批量复现（source-free 两阶段）。

协议（与主表口径一致）：
- 阶段 1：source.py 源域预训练 10 epoch（保存源模型）
- 阶段 2：target.py 目标域 source-free 适应 15 epoch
- 逐轮记录每 epoch 的目标测试集 acc 到 history CSV

用法：
    python -m comparison_experiments.dac.make_lists     # 先生成列表
    python -m comparison_experiments.dac.run_all [--tasks A-B A-C B-C] [--gpu-id 0]
输出：
    comparison_experiments/dac/results_dac.csv    # 汇总（每任务每种子最终 acc）
    comparison_experiments/dac/history_dac.csv    # 每轮记录（phase, round, iter, acc）
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime

# 服务器 Windows 控制台 GBK：打印子进程输出含特殊字符时用替换符，避免 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from comparison_experiments.dac.make_lists import main as make_lists_main

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VISDA_DIR = os.path.join(ROOT, "comparison_experiments", "third_party", "DaC", "VisDA")
SEEDS = [42, 123, 777, 2024, 3407]
# s/t 为 oct 数据集索引：0=BOE, 1=TMI, 2=CELL
TASK_ST = {"A-B": (0, 1), "A-C": (0, 2), "B-C": (1, 2)}
# 官方每轮评估打印：Task: {name}, Iter:{n}/{max}; Accuracy = {x}%. AUC = {y}%
ITER_ACC_RE = re.compile(r"Iter:(\d+)/\d+; Accuracy = ([\d.]+)%. AUC = ([\d.]+)%")
ACC_RE = re.compile(r"Accuracy of the network on the \d+ test images:\s*([\d.]+)%")
ACC_RE2 = re.compile(r"acc:\s*([\d.]+)")

HISTORY_FIELDS = ["task", "seed", "phase", "round", "iter", "acc", "auc"]


def run_stage(cmd, cwd, tag, task, seed, phase, history_rows=None):
    print("\n[{}] cmd: {}".format(tag, " ".join(cmd)))
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    acc = None
    auc = None
    rnd = 0
    for line in proc.stdout:
        line = line.rstrip()
        if ('%|' in line) or ('it/s' in line):
            continue
        if line:
            print(line)
        # 逐轮记录：官方按 epoch 打印 Iter 与 Accuracy/AUC
        m = ITER_ACC_RE.search(line)
        if m and history_rows is not None:
            rnd += 1
            acc = float(m.group(2))
            auc = float(m.group(3))
            history_rows.append({"task": task, "seed": seed, "phase": phase,
                                 "round": rnd, "iter": int(m.group(1)),
                                 "acc": acc, "auc": auc})
        m = ACC_RE.search(line) or ACC_RE2.search(line)
        if m:
            acc = float(m.group(1))
    proc.wait()
    if proc.returncode != 0:
        print("[ERROR] {} 失败 rc={}".format(tag, proc.returncode))
        sys.exit(1)
    return acc, auc


def run_one(task, seed, gpu_id="0", history_rows=None, skip_source=False):
    t0 = datetime.now()
    print("[TIME] DaC {} seed={} start: {}".format(task, seed, t0.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    s, t = TASK_ST[task]
    exp = "experiments/OCT"
    out_src = "{}/source".format(exp)
    out_tgt = "{}/target".format(exp)
    src_ckpt = os.path.join(VISDA_DIR, out_src, "source_F.pt")
    if skip_source and os.path.exists(src_ckpt):
        print("[DaC] 源模型已存在，跳过 source 阶段: {}".format(src_ckpt))
    else:
        # 阶段 1：源域预训练 10 epoch（--trte val 用目标验证集选模型，与官方协议一致）
        src_cmd = [sys.executable, "source.py", "--trte", "val",
                   "--output", out_src, "--da", "uda",
                   "--gpu_id", gpu_id, "--dset", "oct", "--net", "resnet50",
                   "--lr", "1e-3", "--max_epoch", "10",
                   "--s", str(s), "--t", str(t), "--seed", str(seed)]
        run_stage(src_cmd, VISDA_DIR, "DaC source {}".format(task),
                  task, seed, "source", history_rows)
    # 阶段 2：目标域 source-free 适应 15 epoch（官方 run_target.sh 参数 + --max_epoch 15）
    tgt_cmd = [sys.executable, "target.py", "--gpu_id", gpu_id,
               "--output", out_tgt, "--output_src", out_src,
               "--da", "uda", "--dset", "oct", "--net", "resnet50",
               "--lr", "5e-4", "--max_epoch", "15",
               "--s", str(s), "--t", str(t),
               "--cls_par", "0.6", "--lamda_m", "1", "--p_threshold", "0.97",
               "--ent_par", "0.1", "--lamda_ad", "0.3", "--ad_method", "EMMD",
               "--seed", str(seed), "--T", str(seed)]
    acc, auc = run_stage(tgt_cmd, VISDA_DIR, "DaC target {}".format(task),
                         task, seed, "target", history_rows)
    secs = (datetime.now() - t0).total_seconds()
    print("[TIME] DaC {} seed={} end: {} ({} s)".format(task, seed, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), int(secs)), flush=True)
    if acc is None:
        print("[WARN] 未捕获最终 acc，请查 target.py 输出格式")
    return acc, auc, secs


def main():
    ap = argparse.ArgumentParser(description='DaC 批量复现')
    ap.add_argument('--tasks', nargs='+', default=['A-B', 'A-C', 'B-C'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--gpu-id', type=str, default='0')
    ap.add_argument('--skip-source', action='store_true',
                    help='源模型已存在时跳过源域训练（直接跑 target 阶段）')
    args = ap.parse_args()

    # 重新生成列表：避免同步过来的列表残留旧机器绝对路径
    make_lists_main()

    rows = []
    history_rows = []
    for task in args.tasks:
        accs = []
        for seed in args.seeds:
            acc, auc, secs = run_one(task, seed, gpu_id=args.gpu_id,
                                     history_rows=history_rows,
                                     skip_source=args.skip_source)
            accs.append(acc if acc is not None else float('nan'))
            rows.append({"task": task, "seed": seed, "acc": acc,
                         "auc": auc, "seconds": secs})
        valid = [a for a in accs if a == a]
        if valid:
            m = sum(valid) / len(valid)
            sd = (sum((a - m) ** 2 for a in valid) / len(valid)) ** 0.5
            print("== {} DaC = {:.2f}±{:.2f}%".format(task, m, sd))
        else:
            print("== {} DaC = 无有效结果".format(task))

    out_csv = os.path.join(HERE, "results_dac.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "acc", "auc", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print("已写入 {}".format(out_csv))

    hist_csv = os.path.join(HERE, "history_dac.csv")
    with open(hist_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        w.writerows(history_rows)
    print("已写入 {}（{} 轮记录）".format(hist_csv, len(history_rows)))


if __name__ == "__main__":
    main()

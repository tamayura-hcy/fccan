# -*- coding: utf-8 -*-
"""train.py —— 调用 TVT 官方代码（third_party/TVT/main.py）跑单任务单种子。

依赖：
  - 官方预训练权重 checkpoint/ViT-B_16.npz（ImageNet-21K，见 README）
  - apex（或按 README 方案 B 打补丁）

用法：
  python -m comparison_experiments.tvt.train --task A-B --seed 42
输出：
  终端打印官方日志尾部 + "RESULT acc=..." 行（供 run_all 解析）
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

# 服务器 Windows 控制台 GBK：打印子进程输出含特殊字符时用替换符，避免 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TVT_DIR = os.path.join(ROOT, "comparison_experiments", "third_party", "TVT")
SEEDS = [42, 123, 777, 2024, 3407]
TASK_SRC_TGT = {"A-B": ("BOE", "TMI"), "A-C": ("BOE", "CELL"),
                "B-C": ("TMI", "CELL")}
ACC_RE = re.compile(r"Best Accuracy:\s*([\d.]+)")
ACC_RE2 = re.compile(r"Best element-wise Accuracy:\s*([\d.]+)")
BEST_AUC_RE = re.compile(r"Best AUC:\s*([\d.]+)")
# 每 eval_every 步评估一轮：Valid Accuracy / Valid AUC 成对出现
VALID_RE = re.compile(r"Valid Accuracy:\s*([\d.]+)")
VALID_AUC_RE = re.compile(r"Valid AUC:\s*([\d.]+)")


def run_one(task, seed, num_steps=5000, img_size=224, batch=64,
            history_rows=None):
    src, tgt = TASK_SRC_TGT[task]
    lists_dir = os.path.join(HERE, "lists", task)
    name = "{}_s{}".format(task, seed)
    cmd = [sys.executable, "main.py",
           "--name", name,
           "--dataset", "office",          # 仅影响 transform；自定义列表用 office 的通用 transform
           "--source_list", os.path.join(lists_dir, "source_list.txt"),
           "--target_list", os.path.join(lists_dir, "target_list.txt"),
           "--test_list", os.path.join(lists_dir, "test_list.txt"),
           "--num_classes", "3",
           "--model_type", "ViT-B_16",
           "--pretrained_dir", os.path.join(TVT_DIR, "checkpoint", "ViT-B_16.npz"),
           "--num_steps", str(num_steps),
           "--img_size", str(img_size),
           "--train_batch_size", str(batch),
           "--eval_batch_size", str(batch),
           "--seed", str(seed),
           "--output_dir", os.path.join(HERE, "outputs"),
           "--decay_type", "cosine",
           "--warmup_steps", "500",
           "--gradient_accumulation_steps", "1",
           "--local_rank", "-1",
           "--fp16", "--fp16_opt_level", "O2"]
    print("[TVT] task={} seed={} cmd: {}".format(task, seed, " ".join(cmd)))
    t0 = datetime.now()
    print("[TIME] TVT {} seed={} start: {}".format(task, seed, t0.strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    proc = subprocess.Popen(cmd, cwd=TVT_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    best = None
    best_auc = None
    rnd = 0
    pbar_n = 0
    cur_acc = None
    for line in proc.stdout:
        line = line.rstrip()
        if ('%|' in line) or ('it/s' in line):
            pbar_n += 1
            if pbar_n % 100 == 0:
                print("[TVT progress] {} tqdm updates (training in progress)".format(pbar_n), flush=True)
            continue
        if line:
            print(line)
        m = VALID_RE.search(line)
        if m:
            cur_acc = float(m.group(1))
        m = VALID_AUC_RE.search(line)
        if m and history_rows is not None:
            rnd += 1
            history_rows.append({"task": task, "seed": seed, "round": rnd,
                                 "acc": cur_acc, "auc": float(m.group(1))})
        m = ACC_RE.search(line)
        if m:
            best = float(m.group(1))
        else:
            m = ACC_RE2.search(line)
            if m:
                best = float(m.group(1))
        m = BEST_AUC_RE.search(line)
        if m:
            best_auc = float(m.group(1))
    proc.wait()
    if proc.returncode != 0:
        print("[ERROR] TVT {} seed={} 失败 rc={}".format(task, seed, proc.returncode))
        sys.exit(1)
    if best is None:
        print("[WARN] TVT {} seed={} 未捕获到 Best Accuracy（history 仍保留）".format(task, seed))
        return None, None, None
    secs = (datetime.now() - t0).total_seconds()
    print("[TIME] TVT {} seed={} end: {} ({} s)".format(task, seed, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), int(secs)), flush=True)
    print("RESULT task={} seed={} acc={:.4f} auc={:.4f}".format(
        task, seed, best, best_auc if best_auc is not None else float('nan')))
    return best, best_auc, secs


def main():
    ap = argparse.ArgumentParser(description='TVT 单任务单种子')
    ap.add_argument('--task', type=str, default='A-B', choices=['A-B', 'A-C', 'B-C'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-steps', type=int, default=5000)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()
    run_one(args.task, args.seed, num_steps=args.num_steps,
            img_size=args.img_size, batch=args.batch)


if __name__ == "__main__":
    main()

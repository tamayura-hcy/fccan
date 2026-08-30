# -*- coding: utf-8 -*-
"""run_three_parallel.py —— TVT / DaC 两方法串行复现。

16GB 单卡上两个方法显存无法共存，改为**串行**：依次完整跑完
TVT → DaC（每个方法内部仍是 3 任务 × 5 种子）。

每个方法输出：
- results_{tvt,dac}.csv（汇总）
- history_{tvt,dac}.csv（逐轮记录）

协议：
- TVT：单阶段 5000 步（官方 VisDA 协议，每 100 步评估一轮，无源/目标分离）
- DaC：源域 10 epoch + 目标域 15 epoch

用法：
    python -m comparison_experiments.run_three_parallel [--methods tvt dac] [--tvt-batch 16]
日志：
    comparison_experiments/logs/{tvt,dac}.log
"""
import argparse
import os
import subprocess
import sys

# 服务器 Windows 控制台 GBK：打印子进程输出含特殊字符时用替换符，避免 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))  # comparison_experiments/
ROOT = os.path.dirname(HERE)                        # 项目根目录（包可见位置）
LOGS_DIR = os.path.join(HERE, "logs")

SEEDS = [42, 123, 777, 2024, 3407]


def build_cmd(method, gpu_id, tvt_batch=None):
    cmd = [sys.executable, "-m", "comparison_experiments.{}.run_all".format(method),
           "--seeds"] + [str(s) for s in SEEDS]
    # TVT 的 run_all 无 --gpu-id，改用环境变量；DaC 支持 --gpu-id
    if gpu_id is not None and method != "tvt":
        cmd += ["--gpu-id", str(gpu_id)]
    if method == "tvt" and tvt_batch is not None:
        cmd += ["--batch", str(tvt_batch)]
    return cmd


def run_method(method, tag, gpu_id, tvt_batch):
    """串行跑一个方法：输出逐行写日志并转发主控台。"""
    cmd = build_cmd(method, gpu_id, tvt_batch=tvt_batch)
    print("\n[MAIN] ===== 开始 {}: {} =====".format(tag, " ".join(cmd)), flush=True)
    env = dict(os.environ)
    if gpu_id is not None and method == "tvt":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # cwd 必须是项目根目录，否则 `-m comparison_experiments.xxx` 找不到包
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env)
    log_path = os.path.join(LOGS_DIR, method + ".log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n==== {} 启动: {} ====\n".format(tag, " ".join(cmd)))
        for line in proc.stdout:
            line = line.rstrip()
            if ('%|' in line) or ('it/s' in line):
                continue
            if line:
                f.write(line + "\n")
                f.flush()
                print("[{}] {}".format(tag, line), flush=True)
    proc.wait()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="TVT/DaC 两方法串行复现")
    ap.add_argument("--methods", nargs="+", default=["tvt", "dac"],
                    help="按顺序串行运行的方法（默认全部）")
    ap.add_argument("--gpu-tvt", type=str, default=None, help="TVT 用哪张卡（默认继承环境）")
    ap.add_argument("--gpu-dac", type=str, default=None, help="DaC 用哪张卡")
    ap.add_argument("--tvt-batch", type=int, default=16, help="TVT 训练 batch（16GB 单卡建议 16）")
    args = ap.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)
    tags = {"tvt": "TVT", "dac": "DaC"}
    gpus = {"tvt": args.gpu_tvt, "dac": args.gpu_dac}

    failed = []
    for method in args.methods:
        rc = run_method(method, tags[method], gpus[method], args.tvt_batch)
        if rc != 0:
            failed.append(tags[method])
            print("[MAIN] {} 失败 rc={}，继续跑下一个方法".format(tags[method], rc), flush=True)
        else:
            print("[MAIN] {} 完成".format(tags[method]), flush=True)

    print("\n[MAIN] 全部结束。结果文件：", flush=True)
    for method in args.methods:
        r = os.path.join(HERE, method, "results_{}.csv".format(method))
        h = os.path.join(HERE, method, "history_{}.csv".format(method))
        print("  {}: {} | {}".format(method, r, h), flush=True)
    if failed:
        print("[MAIN] 失败方法：{}".format(", ".join(failed)), flush=True)
        sys.exit(1)
    print("[MAIN] 全部成功", flush=True)


if __name__ == "__main__":
    main()

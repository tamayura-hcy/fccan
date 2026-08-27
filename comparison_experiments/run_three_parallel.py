# -*- coding: utf-8 -*-
"""Serial reproduction of TVT / DaC baselines.

They cannot share a 16GB GPU, so they run sequentially (each 3 tasks x 5 seeds).
Output per method: results_{tvt,dac}.csv and history_{tvt,dac}.csv.

Usage:
    python -m comparison_experiments.run_three_parallel [--methods tvt dac] [--tvt-batch 16]
Logs: comparison_experiments/logs/{tvt,dac}.log
"""
import argparse
import os
import subprocess
import sys

# Windows GBK console: use replacement chars for subprocess output to avoid UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))  # comparison_experiments/
ROOT = os.path.dirname(HERE)                        # project root (package-visible location)
LOGS_DIR = os.path.join(HERE, "logs")

SEEDS = [42, 123, 777, 2024, 3407]


def build_cmd(method, gpu_id, tvt_batch=None):
    cmd = [sys.executable, "-m", "comparison_experiments.{}.run_all".format(method),
           "--seeds"] + [str(s) for s in SEEDS]
    # TVT's run_all has no --gpu-id, use env var instead; DaC supports --gpu-id
    if gpu_id is not None and method != "tvt":
        cmd += ["--gpu-id", str(gpu_id)]
    if method == "tvt" and tvt_batch is not None:
        cmd += ["--batch", str(tvt_batch)]
    return cmd


def run_method(method, tag, gpu_id, tvt_batch):
    """Run one method serially: write output line-by-line to log and forward to console."""
    cmd = build_cmd(method, gpu_id, tvt_batch=tvt_batch)
    print("\n[MAIN] ===== Start {}: {} =====".format(tag, " ".join(cmd)), flush=True)
    env = dict(os.environ)
    if gpu_id is not None and method == "tvt":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # cwd must be the project root, otherwise -m comparison_experiments.xxx cannot find the package
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env)
    log_path = os.path.join(LOGS_DIR, method + ".log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n==== {} started: {} ====\n".format(tag, " ".join(cmd)))
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
    ap = argparse.ArgumentParser(description="Serial reproduction of TVT/DaC")
    ap.add_argument("--methods", nargs="+", default=["tvt", "dac"],
                    help="Methods to run serially in order (default: all)")
    ap.add_argument("--gpu-tvt", type=str, default=None, help="GPU for TVT (default: inherit env)")
    ap.add_argument("--gpu-dac", type=str, default=None, help="GPU for DaC")
    ap.add_argument("--tvt-batch", type=int, default=16, help="TVT training batch (suggest 16 on a 16GB GPU)")
    args = ap.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)
    tags = {"tvt": "TVT", "dac": "DaC"}
    gpus = {"tvt": args.gpu_tvt, "dac": args.gpu_dac}

    failed = []
    for method in args.methods:
        rc = run_method(method, tags[method], gpus[method], args.tvt_batch)
        if rc != 0:
            failed.append(tags[method])
            print("[MAIN] {} failed rc={}, continue with the next method".format(tags[method], rc), flush=True)
        else:
            print("[MAIN] {} done".format(tags[method]), flush=True)

    print("\n[MAIN] All finished. Result files:", flush=True)
    for method in args.methods:
        r = os.path.join(HERE, method, "results_{}.csv".format(method))
        h = os.path.join(HERE, method, "history_{}.csv".format(method))
        print("  {}: {} | {}".format(method, r, h), flush=True)
    if failed:
        print("[MAIN] Failed methods: {}".format(", ".join(failed)), flush=True)
        sys.exit(1)
    print("[MAIN] All succeeded", flush=True)


if __name__ == "__main__":
    main()

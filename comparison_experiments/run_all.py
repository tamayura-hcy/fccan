"""Run all baseline methods x tasks x seeds, collect results into results_summary.csv.

Usage:
    python -m comparison_experiments.run_all --methods dann,mcc --tasks A-B,A-C --seeds 777,42
"""
import argparse
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_METHODS = ['dann', 'adda', 'cdan', 'mcc', 'shot', 'svdna', 'emdda']
ALL_TASKS = ['A-B', 'A-C', 'B-C']
DEFAULT_SEEDS = [42, 123, 777, 2024, 3407]


# Official training config per method (consistent with run_comparison.py; --epochs/--batch only a fallback)
METHOD_EPOCHS_BATCH = {
    # thuml TLL official: DANN/CDAN 20ep x 32, MCC 20ep x 36 (batch can be reduced for VRAM)
    'dann':  (20, 32),
    'cdan':  (20, 32),
    'mcc':   (20, 36),
    # SHOT official: source 20ep / target 15ep, batch 64
    'shot':  (15, 64),
    # SVDNA: no official training code (MICCAI'22 repo has only README/Colab); self-implemented
    'svdna': (10, 16),
    # OCT-DDA official ADDA/EM-DDA: 30ep x batch 8 (required, do not change)
    'adda':  (30, 8),
    'emdda': (30, 8),
}


def run_one(method, task, seed, epochs, batch):
    src, tgt = task.split('-')
    out = os.path.join(ROOT, 'comparison_experiments', 'results',
                       '{}__{}__s{}'.format(method, task, seed))
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, 'run.log')
    if method in METHOD_EPOCHS_BATCH:
        epochs, batch = METHOD_EPOCHS_BATCH[method]
    cmd = [sys.executable, '-m',
           'comparison_experiments.{}.train'.format(method),
           '--src', src, '--tgt', tgt, '--seed', str(seed),
           '--epochs', str(epochs), '--batch', str(batch)]
    print("RUN {} {} s{} -> {}  (epochs={} batch={})".format(method, task, seed, log_path, epochs, batch))
    with open(log_path, 'w', encoding='utf-8') as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    # parse last "test acc=" line
    acc = auc = None
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'test acc=' in line:
                try:
                    acc = float(line.split('acc=')[1].split()[0])
                    auc = float(line.split('auc=')[1].split()[0])
                except Exception:
                    pass
    return acc, auc


def main():
    parser = argparse.ArgumentParser(description='Run all comparison baselines')
    parser.add_argument('--methods', type=str, default=','.join(ALL_METHODS))
    parser.add_argument('--tasks', type=str, default=','.join(ALL_TASKS))
    parser.add_argument('--seeds', type=str, default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    methods = [m for m in args.methods.split(',') if m in ALL_METHODS]
    tasks = args.tasks.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]

    rows = []
    for method in methods:
        for task in tasks:
            for seed in seeds:
                acc, auc = run_one(method, task, seed, args.epochs, args.batch)
                rows.append([method, task, seed, acc, auc])

    os.makedirs(os.path.join(ROOT, 'comparison_experiments', 'results'), exist_ok=True)
    csv_path = os.path.join(ROOT, 'comparison_experiments', 'results', 'results_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'task', 'seed', 'acc', 'auc'])
        for r in rows:
            w.writerow(r)
    print("saved -> {}".format(csv_path))

    # mean/std summary
    print("\n=== Summary (mean over seeds) ===")
    for method in methods:
        for task in tasks:
            accs = [r[3] for r in rows if r[0] == method and r[1] == task and r[3] is not None]
            if accs:
                mean = sum(accs) / len(accs)
                var = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
                print("  {:6s} {}  acc={:.4f}±{:.4f}  n={}".format(method, task, mean, var, len(accs)))


if __name__ == '__main__':
    main()

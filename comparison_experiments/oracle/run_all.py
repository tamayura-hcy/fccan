"""跑 oracle 上界：3 任务 × 5 种子，输出 mean±std。

与主表场景对应：
  A->B (BOE->TMI) : oracle = TMI 全监督  → tgt B
  A->C (BOE->CELL): oracle = CELL 全监督 → tgt C
  B->C (TMI->CELL): oracle = CELL 全监督 → tgt C（与 A->C 相同，可复用结果）

Usage:
    python -m comparison_experiments.oracle.run_all          # 全跑（3 任务 × 5 种子）
    python -m comparison_experiments.oracle.run_all --tgt B  # 只跑目标 B
结果写入 comparison_experiments/results/oracle_summary.txt
"""
import argparse
import os
import sys
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.oracle.train import run_oracle, SEEDS

# 场景 → (目标标签, 场景名)
SCENARIOS = [
    ('B', 'A->B'),
    ('C', 'A->C'),
    ('C', 'B->C'),
]
OUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'oracle_summary.txt')


def main():
    ap = argparse.ArgumentParser(description='Oracle 上界：3 任务 × 5 种子')
    ap.add_argument('--tgt', type=str, default=None, choices=['B', 'C'],
                    help='只跑指定目标域（B 或 C），默认跑全部')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--skip-existing', action='store_true',
                    help='跳过已记录的结果')
    args = ap.parse_args()

    results = {tgt: [] for tgt in ('B', 'C')}   # tgt -> [(acc, auc)]
    lines = ["# Oracle 上界（目标域全监督 ResNet-50，5 种子 mean±std）"]
    for tgt, scen in SCENARIOS:
        if args.tgt and tgt != args.tgt:
            continue
        accs, aucs = [], []
        for seed in SEEDS:
            print("\n===== {}  oracle tgt={} seed={} =====".format(scen, tgt, seed))
            acc, auc = run_oracle(tgt, seed, epochs=args.epochs)
            accs.append(acc); aucs.append(auc)
            lines.append("{} tgt={} s{:5d}  acc={:.4f}  auc={:.4f}".format(scen, tgt, seed, acc, auc))
        if len(accs) == 5:
            acc_m, acc_s = statistics.mean(accs), statistics.stdev(accs)
            auc_m, auc_s = statistics.mean(aucs), statistics.stdev(aucs)
            lines.append("==> {} oracle: acc={:.4f}±{:.4f}  auc={:.4f}±{:.4f}".format(
                scen, acc_m, acc_s, auc_m, auc_s))
            results[tgt].append((acc_m, acc_s, auc_m, auc_s))
            print("\n  {} oracle: acc={:.4f}±{:.4f}  auc={:.4f}±{:.4f}".format(
                scen, acc_m, acc_s, auc_m, auc_s))

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print("\n结果已写入 {}".format(OUT_FILE))


if __name__ == '__main__':
    main()

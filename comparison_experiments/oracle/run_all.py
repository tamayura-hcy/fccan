"""Run the oracle upper bound: 3 tasks x 5 seeds, report mean±std.

Scenarios: A->B (BOE->TMI) and A->C/B->C (BOE/TMI -> CELL), oracle = fully
supervised training on the target domain.

Usage:
    python -m comparison_experiments.oracle.run_all          # all (3 tasks x 5 seeds)
    python -m comparison_experiments.oracle.run_all --tgt B  # target B only
Results written to comparison_experiments/results/oracle_summary.txt
"""
import argparse
import os
import sys
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.oracle.train import run_oracle, SEEDS

# scenario -> (target label, scenario name)
SCENARIOS = [
    ('B', 'A->B'),
    ('C', 'A->C'),
    ('C', 'B->C'),
]
OUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'oracle_summary.txt')


def main():
    ap = argparse.ArgumentParser(description='Oracle upper bound: 3 tasks x 5 seeds')
    ap.add_argument('--tgt', type=str, default=None, choices=['B', 'C'],
                    help='Run only the specified target domain (B or C), default all')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip already recorded results')
    args = ap.parse_args()

    results = {tgt: [] for tgt in ('B', 'C')}   # tgt -> [(acc, auc)]
    lines = ["# Oracle upper bound (fully supervised target ResNet-50, 5-seed mean±std)"]
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
    print("\nResults saved to {}".format(OUT_FILE))


if __name__ == '__main__':
    main()

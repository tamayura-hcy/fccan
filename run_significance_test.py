# -*- coding: utf-8 -*-
"""Significance tests: FCCAN vs baselines (3 tasks x 5 seeds).

Reviewer M4: add significance marks to Table 1. Uses per-seed acc with
Welch's t-test + paired Wilcoxon; outputs p-values and significance (p<0.05).
Per-seed acc is read from comparison_experiments/results/; where missing,
Welch is computed from the main table's mean±std (n=5).

Usage: python run_significance_test.py --task AB
Output: comparison_experiments/results/significance.txt
"""
import argparse
import os
import re
import glob
import statistics
from itertools import combinations

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[warn] scipy not installed, fall back to manual Welch t-test")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "comparison_experiments", "results")
OUT_FILE = os.path.join(RESULTS_DIR, "significance.txt")

# Main-table FCCAN per-seed acc (FULL, EMA final, from best/SUMMARY.txt, 2026-08-10).
FCCAN_SEEDS = {
    "AB": [0.9641, 0.9499, 0.9314, 0.9662, 0.9673],   # BOE->TMI
    "AC": [0.8943, 0.8638, 0.8961, 0.8405, 0.8925],   # BOE->CELL
    "BC": [0.9158, 0.9176, 0.9176, 0.9158, 0.9391],   # TMI->CELL
}


def welch_t(x, y):
    """Manual Welch's t-test; return (t, p) two-sided."""
    nx, ny = len(x), len(y)
    mx, my = statistics.mean(x), statistics.mean(y)
    vx, vy = statistics.variance(x), statistics.variance(y)
    se = (vx / nx + vy / ny) ** 0.5
    if se == 0:
        return float('inf'), 0.0 if mx > my else 1.0
    t = (mx - my) / se
    # Welch-Satterthwaite df approximation
    df = (vx / nx + vy / ny) ** 2 / (
        (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
    ) if nx > 1 and ny > 1 else 1.0
    if HAVE_SCIPY:
        p = stats.t.sf(abs(t), df) * 2
    else:
        # Normal approximation without scipy (fine for large df)
        from math import erf, sqrt
        p = 1.0 - erf(abs(t) / sqrt(2.0))
    return t, p


def wilcoxon(x, y):
    """Paired Wilcoxon signed-rank test (two-sided)."""
    if HAVE_SCIPY:
        try:
            return stats.wilcoxon(x, y).pvalue
        except Exception:
            return float('nan')
    return float('nan')


def load_method_seeds(method, task):
    """Read per-seed acc of a method/task from results dir (logs/txt with 'final tgt_acc='). Returns None if missing."""
    pats = [
        os.path.join(RESULTS_DIR, method, "{}_s*.log".format(task)),
        os.path.join(RESULTS_DIR, method, "{}_s*.txt".format(task)),
        os.path.join(RESULTS_DIR, method, "*{}*s*.log".format(task)),
    ]
    accs = []
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            try:
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        m = re.search(r'final tgt_acc=([\d.]+)', line)
                        if m:
                            accs.append(float(m.group(1)))
            except Exception:
                pass
    return accs if len(accs) == 5 else None


# results_comparison_all.csv: per-seed records for CAT/DDC etc.
CSV_PATH = os.path.join(HERE, "results_comparison_all.csv")


def load_method_seeds_csv(method, task):
    """Read per-seed acc (5 seeds) of a method/task from results_comparison_all.csv."""
    if not os.path.exists(CSV_PATH):
        return None
    csv_task = task.replace('AB', 'A-B').replace('AC', 'A-C').replace('BC', 'B-C')
    accs = []
    try:
        with open(CSV_PATH, encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().strip().split(',')
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) < len(header):
                    continue
                rec = dict(zip(header, parts))
                if (rec.get('type') == 'summary'
                        and rec.get('method', '').lower() == method.lower()
                        and rec.get('task') == csv_task):
                    try:
                        accs.append(float(rec['acc']))
                    except (ValueError, KeyError):
                        pass
    except Exception:
        return None
    return accs if len(accs) == 5 else None


# Baselines (Table 1: 3 source-only + 11 DA = 14 unlabeled methods);
# oracle is the supervised reference row, excluded from significance tests.
METHODS = ["ResNet-18", "ResNet-50", "VGG-16",
           "DANN", "ADDA", "CDAN", "EM-DDA", "MCC", "SHOT",
           "CAT", "SVDNA", "DAGCN"]

# Main-table 5-seed mean±std (paper_ieee/04_experiments.tex, percent -> decimal).
# For methods without per-seed data, use Welch on mean±std only (n=5).
METHOD_STATS = {
    # method -> {task: (mean, std)}
    "ResNet-18": {"AB": (0.5394, 0.0320), "AC": (0.6807, 0.0231), "BC": (0.5939, 0.0370)},
    "ResNet-50": {"AB": (0.5203, 0.0248), "AC": (0.6154, 0.0295), "BC": (0.5527, 0.0422)},
    "VGG-16":    {"AB": (0.3603, 0.0608), "AC": (0.3644, 0.0196), "BC": (0.3771, 0.0327)},
    "DANN":      {"AB": (0.7022, 0.0431), "AC": (0.7240, 0.0722), "BC": (0.8161, 0.0192)},
    "CDAN":      {"AB": (0.7723, 0.0239), "AC": (0.7735, 0.0287), "BC": (0.8150, 0.0186)},
    "MCC":       {"AB": (0.6739, 0.0271), "AC": (0.6305, 0.0395), "BC": (0.8147, 0.0245)},
    "SHOT":      {"AB": (0.6255, 0.0935), "AC": (0.5627, 0.0704), "BC": (0.8215, 0.0064)},
    "CAT":       {"AB": (0.8433, 0.0522), "AC": (0.8244, 0.0440), "BC": (0.8527, 0.0197)},
    "SVDNA":     {"AB": (0.4980, 0.0924), "AC": (0.6885, 0.0096), "BC": (0.7545, 0.0481)},
    "DAGCN":     {"AB": (0.8510, 0.0257), "AC": (0.7964, 0.0227), "BC": (0.8875, 0.0224)},
    "ADDA":      {"AB": (0.7020, 0.0538), "AC": (0.6649, 0.0311), "BC": (0.7527, 0.0354)},
    "EM-DDA":    {"AB": (0.8059, 0.0756), "AC": (0.7409, 0.0629), "BC": (0.8946, 0.0299)},
}


def welch_from_stats(mean1, std1, mean2, std2, n1=5, n2=5):
    """Welch independent-sample t-test from mean±std only (n1=n2=5)."""
    se = (std1 ** 2 / n1 + std2 ** 2 / n2) ** 0.5
    if se == 0:
        return float('inf'), 0.0 if mean1 > mean2 else 1.0
    t = (mean1 - mean2) / se
    df = (std1 ** 2 / n1 + std2 ** 2 / n2) ** 2 / (
        (std1 ** 2 / n1) ** 2 / (n1 - 1) + (std2 ** 2 / n2) ** 2 / (n2 - 1)
    )
    if HAVE_SCIPY:
        p = stats.t.sf(abs(t), df) * 2
    else:
        from math import erf, sqrt
        p = 1.0 - erf(abs(t) / sqrt(2.0))
    return t, p


def main():
    ap = argparse.ArgumentParser(description='Significance test: FCCAN vs baselines')
    ap.add_argument('--task', type=str, default=None, choices=['AB', 'AC', 'BC'])
    args = ap.parse_args()

    tasks = ['AB', 'AC', 'BC'] if not args.task else [args.task]
    lines = ["# FCCAN vs baselines significance (5-seed Welch t-test, Holm-Bonferroni correction)",
             "# 17 unlabeled baselines in Table 1 (3 source-only + src_only_FEA-Net + 13 DA) x 3 tasks = 51 groups",
             "# oracle is the supervised reference row, excluded"]
    results = []   # (task, method, p, src, wilcoxon_p, seeds_str)
    for task in tasks:
        fccan = FCCAN_SEEDS.get(task)
        if not fccan:
            lines.append("[warn] {} FCCAN per-seed missing, skip".format(task))
            continue
        fm, fs = statistics.mean(fccan), statistics.stdev(fccan)
        print("\n===== {}  FCCAN mean={:.4f}±{:.4f} seeds={}".format(
            task, fm, fs, fccan))
        for method in METHODS:
            seeds = load_method_seeds(method, task)
            if not seeds:
                seeds = load_method_seeds_csv(method, task)
            if seeds:
                t, p = welch_t(fccan, seeds)
                wp = wilcoxon(fccan, seeds)
                src = "per-seed"
                seeds_str = "/".join("{:.4f}".format(s) for s in seeds)
            else:
                st = METHOD_STATS.get(method, {}).get(task)
                if not st:
                    print("  {:<18s} [no data]".format(method))
                    lines.append("{:4s} {:<18s} [no data]".format(task, method))
                    continue
                t, p = welch_from_stats(fm, fs, st[0], st[1])
                wp = float('nan')
                src = "mean±std"
                seeds_str = "{:.4f}±{:.4f}".format(st[0], st[1])
            results.append((task, method, p, src, wp, seeds_str))

    # Holm-Bonferroni: sort all p ascending, test stepwise.
    n = len(results)
    ordered = sorted(results, key=lambda r: r[2])
    holm = {}
    for k, (task, method, p, src, wp, seeds_str) in enumerate(ordered):
        thr = 0.05 / (n - k)          # threshold for the (k+1)-th smallest
        holm[(task, method)] = (p <= thr)
    n_sig = 0
    lines.append("")
    lines.append("# ===== Holm-Bonferroni ({} groups, p ascending) =====".format(n))
    lines.append("# significant = p <= 0.05/(n-k+1) and p < 0.05")
    for task, method, p, src, wp, seeds_str in results:
        sig = "sig" if holm[(task, method)] else "ns"
        n_sig += holm[(task, method)]
        print("  {:<18s} [{}] p={:.4g} Holm={}  wilcoxon_p={:.4g}  seeds={}"
              .format(method, src, p, sig, wp, seeds_str))
        lines.append("{:4s} {:<18s} [{}] p={:.4g} {} wilcoxon_p={:.4g} seeds={}".format(
            task, method, src, p, sig, wp, seeds_str))

    print("\nHolm-significant {}/{}".format(n_sig, n))
    lines.append("")
    lines.append("==> Holm-significant {}/{}".format(n_sig, n))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print("\nwritten to {}".format(OUT_FILE))



if __name__ == '__main__':
    main()

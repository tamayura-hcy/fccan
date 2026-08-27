#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified ablation runner (main 8 w/o modules + extra 4 groups).

Main (--phase main): 8 w/o-module ablations, reporting the 10 full metrics of the
final-epoch EMA teacher; results: results_ablation_3tasks.txt.
Extra (--phase extra): no_ema / no_fea_ll / oracle / src_only_fea, reporting acc/auc;
results: results_ablation_extra.txt.

Task configs (best.md): AB es=5 et=8; AC es=4 et=15; BC es=8 et=15.
Usage: python run_ablation.py --phase main|extra|all --tasks AB,AC --plans no_fea,no_em
"""
import subprocess, sys, os, time, re, statistics, argparse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

SEEDS = [42, 123, 777, 2024, 3407]
ITERS = 1
CELL_SPLIT = "CELL_split_2025"
TIMEOUT = 4 * 3600

# Full 10-metric line parser
TEST_METRICS_RE = re.compile(
    r'test acc=(-?[\d.]+) \| auc=(-?[\d.]+) \| recall=(-?[\d.]+) \| precision=(-?[\d.]+) \| f1=(-?[\d.]+)'
    r' \| bacc=(-?[\d.]+) \| specificity=(-?[\d.]+) \| kappa=(-?[\d.]+) \| gmean=(-?[\d.]+) \| mcc=(-?[\d.]+)')
METRIC_KEYS = ['acc', 'auc', 'recall', 'precision', 'f1', 'bacc',
               'specificity', 'kappa', 'gmean', 'mcc']

# Task best configs (keys AB/AC/BC)
TASKS = {
    "AB": {"name": "BOE->TMI", "es": 5, "et": 8,
           "extra": ["--src_ll_prob", "0.7", "--src_ll_alpha", "2.0"]},
    "AC": {"name": "BOE->CELL", "es": 4, "et": 15,
           "extra": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.001",
                     "--alpha_scon", "0.01"]},
    "BC": {"name": "TMI->CELL", "es": 8, "et": 15,
           "extra": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.5"]},
}

# ---- main phase: 8 w/o plans (10 full metrics) ----
MAIN_RESULTS_FILE = os.path.join(HERE, "results_ablation_3tasks.txt")
MAIN_LOG_DIR = os.path.join(HERE, "logs_ablation_3tasks")
MAIN_PLANS = [
    ("no_fea",  ["--use_fea_net", "0"],                       "w/o FEA backbone (incl. HFComp)"),
    ("no_em",   ["--lambda_em", "0"],                         "w/o EM entropy minimization"),
    ("no_ll",   ["--src_ll_aug", "0", "--scw_ll", "0",
                  "--lambda_llinv", "0"],                     "w/o LL group (src LL aug + SCW-LL + L_llinv)"),
    ("no_caco", ["--lambda_caco", "0"],                       "w/o CaCo category contrast"),
    ("no_scon", ["--alpha_scon", "0"],                        "w/o SCON energy normalization"),
    ("no_ang",  ["--lambda_batch_ang", "0"],                  "w/o ANG angular balancing"),
    ("no_ema",  ["--ema_guide_caco", "0"],                    "w/o EMA teacher guidance"),
    ("no_src",  ["--lambda_src", "0"],                        "w/o source classification constraint"),
]


def _parse_metrics(line):
    m = TEST_METRICS_RE.search(line)
    if not m:
        return None
    return {k: float(v) for k, v in zip(METRIC_KEYS, m.groups())}


def _run_proc(cmd, cwd, log_path, skip_existing, timeout=TIMEOUT):
    """Run a subprocess; return the full log text (None if skipped)."""
    if skip_existing and os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        print("[skip] (log exists) {}".format(" ".join(cmd[1:])))
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print("\n[RUN]  {}".format(" ".join(cmd[1:])))
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    lines = []
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace",
                                env=env, cwd=cwd)
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            is_tqdm = ('%|' in line) or ('it/s' in line)
            if not is_tqdm and line:
                print(line)
        proc.wait(timeout=timeout)
    except Exception:
        pass
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
    text = "\n".join(lines) + "\n"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def _main_final_ema_from_log(log_path):
    """Parse final-epoch EMA metrics from a main-phase log (fall back to Student)."""
    ema_metrics = None
    last_m = None
    student_metrics = None
    in_target_test = False
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                _m = _parse_metrics(line)
                if _m is not None:
                    last_m = _m
                if "Test tgt_encoder + classifier on Target Test dataset" in line:
                    in_target_test = True
                    continue
                if in_target_test:
                    if _m is not None:
                        student_metrics = _m
                    elif 'epoch' in line or '===' in line:
                        in_target_test = False
                if "ema_acc=" in line:
                    try:
                        val = float(line.split("ema_acc=")[1].split()[0].strip())
                        mtr = dict(last_m) if last_m else None
                        if mtr is not None:
                            mtr['acc'] = val
                        ema_metrics = mtr
                    except Exception:
                        pass
    except Exception:
        pass
    return ema_metrics if ema_metrics is not None else student_metrics


def run_main_phase(task_keys, plan_tags, seeds, skip_existing):
    """Main ablation: 8 w/o plans x tasks x seeds; report final-epoch EMA 10 metrics."""
    def _write(text):
        with open(MAIN_RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(text)
            f.flush()

    plans = [p for p in MAIN_PLANS if p[0] in plan_tags] if plan_tags else MAIN_PLANS
    os.makedirs(MAIN_LOG_DIR, exist_ok=True)
    _write("# main ablation (8 modules removed, final EMA) start {}\n".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    for tk in task_keys:
        t = TASKS[tk]
        _write("# task {} {} (es={}/et={})  plans={}  seeds={}\n".format(
            tk, t["name"], t["es"], t["et"], [p[0] for p in plans], seeds))
        for tag, overrides, desc in plans:
            accs = []
            for seed in seeds:
                out_dir = os.path.join(MAIN_LOG_DIR, "{}__{}__s{}".format(tag, tk, seed))
                log_path = os.path.join(out_dir, "run.log")
                cmd = [sys.executable, "main.py", "--only", t["name"],
                       "-es", str(t["es"]), "-et", str(t["et"]),
                       "--seed", str(seed), "-i", str(ITERS),
                       "--cell_split", CELL_SPLIT] + t["extra"] + overrides
                _run_proc(cmd, HERE, log_path, skip_existing)
                mtr = _main_final_ema_from_log(log_path)
                if mtr is not None:
                    accs.append(mtr['acc'])
                    _ms = " ".join("{}={:.4f}".format(k, mtr.get(k, 0.0)) for k in METRIC_KEYS)
                    _write("{:8s} {:4s} s{:5d}  EMA-epoch-{} acc={:.4f}  {}\n".format(
                        tag, tk, seed, t["et"], mtr['acc'], _ms))
                else:
                    _write("{:8s} {:4s} s{:5d}  EMA=NA(no ema_acc in log)\n".format(tag, tk, seed))
            if len(accs) >= 2:
                _write("  -> {} {}  mean={:.4f}±{:.4f}  n={}\n".format(
                    tag, tk, statistics.mean(accs), statistics.stdev(accs), len(accs)))
            elif accs:
                _write("  -> {} {}  acc={:.4f} n=1\n".format(tag, tk, accs[0]))
    _write("# main ablation end {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("\n=== main ablation done: results in {} ===".format(MAIN_RESULTS_FILE))


# ---- extra phase: 4 supplementary groups (acc/auc) ----
EXTRA_RESULTS_FILE = os.path.join(HERE, "results_ablation_extra.txt")
EXTRA_LOG_DIR = os.path.join(HERE, "logs_ablation_extra")
EXTRA_GROUPS = {
    "no_ema": ["--ema_teacher", "0", "--ema_guide_caco", "0", "--final_metric", "student"],
    "no_fea_ll": ["--use_fea_net", "0", "--src_ll_aug", "0", "--scw_ll", "0", "--lambda_llinv", "0"],
    "src_only_fea": ["--transferlearning", "0"],
}
ORACLE_TGTS = {"B": ["AB"], "C": ["AC", "BC"]}
ORACLE_RE = re.compile(r'\[Oracle\] final tgt_acc=([\d.]+) tgt_auc=([\d.]+)')
NOTRANSFER_RE = re.compile(r'No transferrring test_acc = ([\d.]+)')


def _extra_parse_result(group, log_text):
    if group == "oracle":
        m = ORACLE_RE.search(log_text)
        if m:
            return float(m.group(1)), float(m.group(2))
        return None, None
    if group == "src_only_fea":
        m = NOTRANSFER_RE.search(log_text)
        if m:
            return float(m.group(1)), None
    acc = auc = None
    for m in TEST_METRICS_RE.finditer(log_text):
        acc, auc = float(m.group(1)), float(m.group(2))
    return acc, auc


def _extra_build_cmd(group, tk, seed):
    if group == "oracle":
        tgt = "B" if tk == "AB" else "C"
        return ([sys.executable, "-m", "comparison_experiments.oracle.train",
                 "--tgt", tgt, "--seed", str(seed)], HERE)
    cfg = TASKS[tk]
    cmd = [sys.executable, "main.py", "--only", cfg["name"],
           "-es", str(cfg["es"]), "--seed", str(seed), "-i", "1",
           "--cell_split", CELL_SPLIT]
    if group != "src_only_fea":
        cmd += ["-et", str(cfg["et"])]
    cmd += cfg["extra"] + EXTRA_GROUPS[group]
    return (cmd, HERE)


def run_extra_phase(group_tags, task_keys, seeds, skip_existing, dry_run):
    """Supplementary ablation: no_ema / no_fea_ll / oracle / src_only_fea; report acc/auc."""
    def _write(text):
        with open(EXTRA_RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(text)
            f.flush()

    groups = [g for g in group_tags if g in EXTRA_GROUPS or g == "oracle"]
    plan = []  # (group, tk, seed)
    for g in groups:
        if g == "oracle":
            for tgt in ("B", "C"):
                if any(t in ORACLE_TGTS[tgt] for t in task_keys):
                    for s in seeds:
                        plan.append((g, ORACLE_TGTS[tgt][0], s))
        else:
            for tk in task_keys:
                for s in seeds:
                    plan.append((g, tk, s))
    print("extra plan {} runs | groups={} tasks={} seeds={}".format(len(plan), groups, task_keys, seeds))

    if dry_run:
        for g, tk, s in plan:
            cmd, _ = _extra_build_cmd(g, tk, s)
            print("[DRY] {}".format(" ".join(cmd[1:])))
        return

    _write("# extra ablation start {} | groups={} tasks={} seeds={}\n".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), groups, task_keys, seeds))
    _write("{:12s} {:4s} {:5s} {:>9s} {:>9s} {:>6s}\n".format("group", "task", "seed", "acc", "auc", "time(s)"))
    results = []
    for g, tk, s in plan:
        cmd, cwd = _extra_build_cmd(g, tk, s)
        # Keep the legacy log dir layout of run_ablation_extra.py
        log_path = os.path.join(EXTRA_LOG_DIR, g, tk, "s{}".format(s), "run.log")
        if skip_existing and os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            text = open(log_path, encoding="utf-8", errors="ignore").read()
            acc, auc = _extra_parse_result(g, text)
            print("[skip] {} {} s{}".format(g, tk, s))
        else:
            t0 = time.time()
            text = _run_proc(cmd, cwd, log_path, False)
            acc, auc = _extra_parse_result(g, text)
            _write("{:12s} {:4s} {:5d}  {}  {}  {:.0f}\n".format(
                g, tk, s,
                "{:.4f}".format(acc) if acc is not None else "NA",
                "{:.4f}".format(auc) if auc is not None else "NA",
                time.time() - t0))
        results.append((g, tk, acc, auc))

    # Summary (mean±std)
    _write("\n=== summary (mean±std over seeds) ===\n")
    for g in groups:
        for tk in task_keys:
            accs = [a for g2, t2, a, _ in results if g2 == g and t2 == tk and a is not None]
            if not accs:
                continue
            m = sum(accs) / len(accs)
            sd = (sum((a - m) ** 2 for a in accs) / len(accs)) ** 0.5 if len(accs) > 1 else 0.0
            _write("  {:12s} {:4s}  acc={:.4f}±{:.4f}  n={}\n".format(g, tk, m, sd, len(accs)))
    _write("# extra ablation end {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("\n=== extra ablation done: results in {} ===".format(EXTRA_RESULTS_FILE))


def main():
    ap = argparse.ArgumentParser(description="Unified ablation runner: main (8 modules) + extra (4 groups)")
    ap.add_argument("--phase", type=str, default="all", choices=["main", "extra", "all"],
                    help="main=8 w/o modules; extra=4 groups; all=both")
    ap.add_argument("--tasks", type=str, default="AB,AC,BC", help="task subset (AB/AC/BC)")
    ap.add_argument("--plans", type=str, default=None, help="main-phase plan subset (no_fea/no_em/...)")
    ap.add_argument("--groups", type=str, default="no_ema,no_fea_ll,oracle,src_only_fea",
                    help="extra-phase group subset")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--dry-run", action="store_true", help="print commands only in the extra phase")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs whose log already exists")
    args = ap.parse_args()

    task_keys = [t.strip() for t in args.tasks.split(",") if t.strip() in TASKS]
    if not task_keys:
        print("--tasks only supports AB/AC/BC")
        return
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    plan_tags = [p.strip() for p in args.plans.split(",") if p.strip()] if args.plans else None

    if args.phase in ("main", "all"):
        run_main_phase(task_keys, plan_tags, seeds, args.skip_existing)
    if args.phase in ("extra", "all"):
        group_tags = [g.strip() for g in args.groups.split(",") if g.strip()]
        run_extra_phase(group_tags, task_keys, seeds, args.skip_existing, args.dry_run)


if __name__ == "__main__":
    main()

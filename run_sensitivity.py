#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-parameter sensitivity scan for A-B / A-C / B-C (unified 3-task entry).

Each run changes one hyperparameter on top of the task's best config; reports the
10 full metrics of the final-epoch (et) EMA teacher. 5 values per parameter,
5 seeds; total 3 x 7 x 5 x 5 = 525 runs.

Task best configs (best.md): AB es=5 et=8; AC es=4 et=15; BC es=8 et=15.
Usage:
    python run_sensitivity.py --tasks AB --params caco,ang --skip-existing
"""
import subprocess, sys, os, time, re, statistics, argparse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

SEEDS = [42, 123, 777, 2024, 3407]
ITERS = 1
CELL_SPLIT = "CELL_split_2025"
TIMEOUT = 4 * 3600

TEST_METRICS_RE = re.compile(
    r'test acc=(-?[\d.]+) \| auc=(-?[\d.]+) \| recall=(-?[\d.]+) \| precision=(-?[\d.]+) \| f1=(-?[\d.]+)'
    r' \| bacc=(-?[\d.]+) \| specificity=(-?[\d.]+) \| kappa=(-?[\d.]+) \| gmean=(-?[\d.]+) \| mcc=(-?[\d.]+)')
METRIC_KEYS = ['acc', 'auc', 'recall', 'precision', 'f1', 'bacc',
               'specificity', 'kappa', 'gmean', 'mcc']

# Task definitions: (key, --only, es, et, BASE_ARGS, desc)
TASKS = {
    "AB": {
        "name": "BOE->TMI", "es": 5, "et": 8,
        "base": ["--src_ll_prob", "0.7", "--src_ll_alpha", "2.0"],
        "desc": "A-B best (best.md, P070A20)",
    },
    "AC": {
        "name": "BOE->CELL", "es": 4, "et": 15,
        "base": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.001", "--alpha_scon", "0.01"],
        "desc": "A-C best (best.md)",
    },
    "BC": {
        "name": "TMI->CELL", "es": 8, "et": 15,
        "base": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.5"],
        "desc": "B-C best (best.md)",
    },
}

# Scan params: (key, flag, paper symbol, best/default, [5 values], desc)
# Cross-env consistency (2026-08-12, user decision): sensitivity runs on a group
# machine with an older PyTorch; the lambda_em=1 tier reuses the local main
# experiment results instead of the group-machine run.
PARAMS = [
    ("caco", "--lambda_caco",      r"$\lambda_{\mathrm{caco}}$", "0.1", [0.0001, 0.001, 0.01, 0.1, 1],   "category contrast weight"),
    ("ang",  "--lambda_batch_ang", r"$\lambda_{\mathrm{ang}}$",  "0.5", [0.0005, 0.005, 0.05, 0.5, 1],  "batch angular balancing weight"),
    ("scon", "--alpha_scon",       r"$\lambda_{\mathrm{scon}}$", "0.1", [0.0001, 0.001, 0.01, 0.1, 1],   "energy normalization weight"),
    ("em",   "--lambda_em",        r"$\lambda_{\mathrm{em}}$",   "1.0", [0.001, 0.01, 0.1, 1, 10],       "entropy minimization weight (engine, can be >1)"),
    ("aug",  "--src_ll_alpha",     r"$\alpha_{\mathrm{aug}}$",   "2.0", [0.2, 0.5, 1, 2, 5],            "LL perturbation gain (best=2, can be >1)"),
    ("src",  "--lambda_src",       r"$\lambda_{\mathrm{src}}$",  "0.1", [0.0001, 0.001, 0.01, 0.1, 1],   "source classification weight"),
    ("wrb",  "--wrb_alpha",        r"$\alpha_{\mathrm{wrb}}$",   "0.4", [0.0001, 0.001, 0.01, 0.1, 0.5], "WRB subband gating strength"),
]
PARAM_KEYS = [p[0] for p in PARAMS]

# When a scanned value equals the task's best value, reuse the local main-experiment result.
BASE_VALUES = {
    "AB": {"caco": 0.1, "ang": 0.5, "scon": 0.1, "em": 1.0, "aug": 2.0, "src": 0.1, "wrb": 0.4},
    "AC": {"caco": 0.01, "ang": 0.001, "scon": 0.01, "em": 1.0, "aug": 1.0, "src": 0.1, "wrb": 0.4},
    "BC": {"caco": 0.01, "ang": 0.5, "scon": 0.1, "em": 1.0, "aug": 1.0, "src": 0.1, "wrb": 0.4},
}
# 5-seed EMA final acc per task from best/SUMMARY.txt (local env, 2026-08-10).
BEST_OVERRIDE = {
    "AB": {42: 0.9641, 123: 0.9499, 777: 0.9314, 2024: 0.9662, 3407: 0.9673},
    "AC": {42: 0.8943, 123: 0.8638, 777: 0.8961, 2024: 0.8405, 3407: 0.8925},
    "BC": {42: 0.9158, 123: 0.9176, 777: 0.9176, 2024: 0.9158, 3407: 0.9391},
}


def is_best_override(task_key, key, value):
    """True when the scanned value equals the task's best value (skip run, reuse best)."""
    bv = BASE_VALUES.get(task_key, {})
    return key in bv and abs(float(value) - bv[key]) < 1e-9


def _results_file(task_key):
    return os.path.join(HERE, "results_sensitivity_{}.txt".format(task_key))


def _log_dir(task_key):
    return os.path.join(HERE, "logs_sensitivity_{}".format(task_key))


def _write(task_key, text):
    with open(_results_file(task_key), "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()


def _parse_metrics(line):
    m = TEST_METRICS_RE.search(line)
    if not m:
        return None
    return {k: float(v) for k, v in zip(METRIC_KEYS, m.groups())}


def _final_ema_from_log(log_path):
    """Parse final-epoch EMA full metrics from an existing log."""
    ema_metrics = None
    last_m = None
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                _m = _parse_metrics(line)
                if _m is not None:
                    last_m = _m
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
    return ema_metrics


def run_one(task_key, param_key, value, seed, skip_existing):
    """Run main.py with one param set to value on top of the task base; report et-epoch EMA.

    Cross-env consistency (2026-08-13): if value equals the task's best value,
    skip the run and reuse the local best/SUMMARY.txt EMA acc for that seed.
    """
    tcfg = TASKS[task_key]
    tag = "{}={}".format(param_key, value)
    if is_best_override(task_key, param_key, value):
        acc = BEST_OVERRIDE.get(task_key, {}).get(seed)
        if acc is not None:
            print("[override] {} {} s{} -> reuse local best acc={:.4f}".format(task_key, tag, seed, acc))
            return {"acc": acc, "auc": None, "recall": None, "precision": None, "f1": None,
                    "bacc": None, "specificity": None, "kappa": None, "gmean": None, "mcc": None}
        print("[override] {} {} s{} -> seed missing in best table, run it".format(task_key, tag, seed))

    out_dir = os.path.join(_log_dir(task_key), "{}__s{}".format(tag, seed))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    if skip_existing and os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        print("[skip] {} {} s{} (log exists)".format(task_key, tag, seed))
        return _final_ema_from_log(log_path)

    flag = [p[1] for p in PARAMS if p[0] == param_key][0]
    # If the scanned flag is already in base, drop it from base to avoid duplication.
    base = tcfg["base"]
    if flag in base:
        _i = base.index(flag)
        base = base[:_i] + base[_i + 2:]
    cmd = [sys.executable, "main.py", "--only", tcfg["name"],
           "-es", str(tcfg["es"]), "-et", str(tcfg["et"]),
           "--seed", str(seed), "-i", str(ITERS),
           "--cell_split", CELL_SPLIT] + base + [flag, str(value)]
    print("\n[RUN] {} {} s{}  {}={}  cmd={}".format(
        task_key, param_key, seed, flag, value, " ".join(cmd[2:])))
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    last_m = None
    ep_metrics = {}
    proc = None
    lines_out = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace",
                                env=env)
        for line in proc.stdout:
            line = line.rstrip()
            lines_out.append(line)
            is_tqdm = ('%|' in line) or ('it/s' in line)
            if not is_tqdm and line:
                print(line)
            _m = _parse_metrics(line)
            if _m is not None:
                last_m = _m
            if "ema_acc=" in line:
                try:
                    m_ep = re.search(r'epoch\s*(\d+)/', line)
                    ep = int(m_ep.group(1)) if m_ep else (len(ep_metrics) + 1)
                    val_ema = float(line.split("ema_acc=")[1].split()[0].strip())
                    mtr = dict(last_m) if last_m else None
                    if mtr is not None:
                        mtr['acc'] = val_ema
                    ep_metrics[ep] = mtr
                except Exception:
                    pass
        proc.wait(timeout=TIMEOUT)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + "\n")
        elapsed = time.time() - t0
        mtr = ep_metrics.get(tcfg["et"])
        if mtr is None and ep_metrics:
            mtr = ep_metrics[max(ep_metrics.keys())]
        if mtr is not None:
            _ms = " ".join("{}={:.4f}".format(k, mtr.get(k, 0.0)) for k in METRIC_KEYS)
            _write(task_key, "{:10s} s{:5d} {}={:<8}  EMA-epoch-{} acc={:.4f}  {}  time={:.0f}s\n".format(
                param_key, seed, flag, value, tcfg["et"], mtr['acc'], _ms, elapsed))
        else:
            _write(task_key, "{:10s} s{:5d} {}={:<8}  EMA=NA(no ema_acc in log)  time={:.0f}s\n".format(
                param_key, seed, flag, value, elapsed))
        return mtr
    except Exception as e:
        print("[ERROR] {} {} s{}: {}".format(task_key, param_key, seed, e))
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines_out) + "\n")
        except Exception:
            pass
        return None
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()


def main():
    ap = argparse.ArgumentParser(description="A-B/A-C/B-C sensitivity analysis (single-param scan, final-epoch EMA)")
    ap.add_argument("--tasks", type=str, default="AB,AC,BC",
                    help="task subset (AB/AC/BC, comma-separated), default all")
    ap.add_argument("--params", type=str, default=None,
                    help="param subset (caco/ang/scon/em/aug/src/wrb, comma-separated), default all")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)),
                    help="seed subset (comma-separated)")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs whose log already exists")
    args = ap.parse_args()

    task_keys = [t.strip() for t in args.tasks.split(",") if t.strip() in TASKS]
    if not task_keys:
        print("--tasks only supports AB,AC,BC (given {})".format(args.tasks))
        return
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    params = PARAMS if not args.params else \
             [p for p in PARAMS if p[0] in [x.strip() for x in args.params.split(",")]]

    for tk in task_keys:
        os.makedirs(_log_dir(tk), exist_ok=True)
        tcfg = TASKS[tk]
        n_total = len(params) * len(params[0][4]) * len(seeds)
        _write(tk, "# {} sensitivity (single-param scan, final EMA) start {} | {} (es={}/et={}, {}) | params={} seeds={} total {} runs\n".format(
            tk, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tcfg["name"],
            tcfg["es"], tcfg["et"], tcfg["desc"], [p[0] for p in params], seeds, n_total))

        results = {p[0]: {v: [] for v in p[4]} for p in params}
        done = 0
        for key, flag, sym, best, vals, desc in params:
            for v in vals:
                for seed in seeds:
                    done += 1
                    m = run_one(tk, key, v, seed, args.skip_existing)
                    if m is not None:
                        results[key][v].append(m)

        # ---- Summary (final EMA acc mean±std) ----
        _write(tk, "\n=== summary (final EMA acc, 5-seed mean±std; et={}) ===\n".format(tcfg["et"]))
        for key, flag, sym, best, vals, desc in params:
            _write(tk, "# param {} {}  best={}  ({})\n".format(sym, flag, best, desc))
            for v in vals:
                accs = [m['acc'] for m in results[key][v]]
                if len(accs) >= 2:
                    mean = statistics.mean(accs)
                    std = statistics.stdev(accs)
                    tag = "  [from local main-experiment best]" if is_best_override(tk, key, v) else ""
                    _write(tk, "  {}={:<8} {:.4f}±{:.4f}  n={}{}\n".format(flag, v, mean, std, len(accs), tag))
                elif len(accs) == 1:
                    tag = "  [from local main-experiment best]" if is_best_override(tk, key, v) else ""
                    _write(tk, "  {}={:<8} {:.4f}         n=1{}\n".format(flag, v, accs[0], tag))
                else:
                    _write(tk, "  {}={:<8} NA                    n=0\n".format(flag, v))

        _write(tk, "# {} sensitivity end {}\n".format(tk, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print("\n=== {} done: results in {} ===".format(tk, _results_file(tk)))


if __name__ == "__main__":
    main()

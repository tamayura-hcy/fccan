#!/usr/bin/env python3
"""Batch experiment template.

Copy and edit BASE/CONFIGS to run many configs x seeds at once.
Usage: python run_xxx.py

Before submitting (checklist):
  1. every --arg exists in main.py's argparse
  2. tqdm lines are filtered (%| or it/s)
  3. every epoch is flushed immediately
  4. -et/-es values are correct
  5. no nohup/& in docstrings
"""
import subprocess, sys, os, time, shutil, traceback, re, statistics
from datetime import datetime

# ---- Per-epoch full-metric parser (test() prints 10 metrics) ----
TEST_METRICS_RE = re.compile(
    r'test acc=(-?[\d.]+) \| auc=(-?[\d.]+) \| recall=(-?[\d.]+) \| precision=(-?[\d.]+) \| f1=(-?[\d.]+)'
    r' \| bacc=(-?[\d.]+) \| specificity=(-?[\d.]+) \| kappa=(-?[\d.]+) \| gmean=(-?[\d.]+) \| mcc=(-?[\d.]+)')


def _parse_test_metrics(line):
    m = TEST_METRICS_RE.search(line)
    if not m:
        return None
    keys = ['acc', 'auc', 'recall', 'precision', 'f1', 'bacc', 'specificity', 'kappa', 'gmean', 'mcc']
    return {k: float(v) for k, v in zip(keys, m.groups())}


def _append_line(results_file, text):
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()

# ============================================================
# EDIT: results file name
# ============================================================
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")

# ============================================================
# EDIT: seed list
# ============================================================
SEEDS = [42, 123]

# ============================================================
# EDIT: timeout (seconds)
# ============================================================
TIMEOUT = 5400

# ============================================================
# EDIT: common base args
# ============================================================
BASE = [
    "--only", "BOE->TMI",
    "-es", "4", "-et", "30",
    # ... your args
]

# ============================================================
# EDIT: experiment configs
# ============================================================
CONFIGS = []
cid = 0

def cfg(name, overrides, seeds=None):
    global cid; cid += 1
    CONFIGS.append((f"{cid:02d}_{name}", overrides, seeds if seeds is not None else SEEDS))

# --- examples ---
cfg("baseline", [])
cfg("exp_01", ["--some_arg", "value"])

# ============================================================
# ---- No changes needed below ----
# ============================================================
TARGET_ITER = 1
N_SEEDS_TOTAL = sum(len(s) for _, _, s in CONFIGS)


def run_one(name, overrides, seed, done, total, results_file):
    tag = f"{name}_s{seed}"
    cmd_args = [sys.executable, "main.py"] + BASE + overrides + [
        "--seed", str(seed), "-i", str(TARGET_ITER)
    ]
    t0 = time.time()
    save_dir = f"saves/FEANet_BOE_to_TMI_iter{TARGET_ITER}"
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)

    # Per-epoch accuracy of student and EMA teacher: {epoch: acc}
    stu_accs = {}   # [Target test] student_acc
    ema_accs = {}   # [EMA test] ema_acc (needs ema_teacher/ema_guide_caco)
    stu_metrics = {}   # epoch -> student full-metric dict
    ema_metrics = {}   # epoch -> EMA full-metric dict
    swa_acc = None     # SWA final acc (when --swa 1)
    swa_metrics = None # SWA final full metrics
    last_m = None      # latest full-metric line
    proc = None
    try:
        proc = subprocess.Popen(
            cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace"
        )
        for line in proc.stdout:
            line = line.rstrip()
            is_tqdm = ('%|' in line) or ('it/s' in line)
            if not is_tqdm and line:
                print(line)
            # Capture full-metric lines (latest = candidate for student/EMA/SWA)
            _m = _parse_test_metrics(line)
            if _m is not None:
                last_m = _m
            # Capture student_acc + full metrics
            if "student_acc=" in line:
                try:
                    m_ep = re.search(r'epoch\s*(\d+)/', line)
                    ep = int(m_ep.group(1)) if m_ep else (len(stu_accs) + 1)
                    stu_accs[ep] = float(line.split("student_acc=")[1].split()[0].strip())
                    if last_m is not None:
                        stu_metrics[ep] = dict(last_m)
                        _ms = " ".join("{}={:.4f}".format(k, last_m[k]) for k in
                                       ['acc', 'auc', 'recall', 'precision', 'f1', 'bacc', 'specificity', 'kappa', 'gmean', 'mcc'])
                        _append_line(results_file, f"{tag:55s} epoch={ep:2d}  STUDENT {_ms}\n")
                except:
                    pass
            # Capture ema_acc + full metrics
            if "ema_acc=" in line:
                try:
                    m_ep = re.search(r'epoch\s*(\d+)/', line)
                    ep = int(m_ep.group(1)) if m_ep else (len(ema_accs) + 1)
                    ema_accs[ep] = float(line.split("ema_acc=")[1].split()[0].strip())
                    if last_m is not None:
                        ema_metrics[ep] = dict(last_m)
                        _ms = " ".join("{}={:.4f}".format(k, last_m[k]) for k in
                                       ['acc', 'auc', 'recall', 'precision', 'f1', 'bacc', 'specificity', 'kappa', 'gmean', 'mcc'])
                        _append_line(results_file, f"{tag:55s} epoch={ep:2d}  EMA {_ms}\n")
                except:
                    pass
            # Capture SWA final (when --swa 1; the preceding metric line is SWA)
            if "swa_acc=" in line and "[SWA test]" in line:
                try:
                    swa_acc = float(line.split("swa_acc=")[1].split()[0].strip())
                    if last_m is not None:
                        swa_metrics = dict(last_m)
                except:
                    pass
            # Capture DiagV2 lines; merge acc + all diagnostics (split by " | ", parse key=val)
            if "DIAG ep" in line:
                try:
                    m = re.search(r'DIAG ep(\d+)', line)
                    if m:
                        ep = int(m.group(1))
                        acc = stu_accs.get(ep, 0.0)
                        ema_acc = ema_accs.get(ep, 0.0)
                        diag_info = line[line.index("DIAG ep"):]
                        # logit(...) nested keys survive the " | " split
                        # fields: rec/pcos_max/fnorm/wnorm/logit/F/etfDev/cacoQ/pdrift/coh/bflip
                        parts = [f"acc={acc:.4f}", f"ema_acc={ema_acc:.4f}"]
                        for seg in diag_info.split('|'):
                            seg = seg.strip()
                            if not seg or seg.startswith('DIAG ep'):
                                continue
                            km = re.match(r'(\w[\w\[\]()]*)\s*=\s*(.*)$', seg)
                            if not km:
                                continue
                            k, v = km.group(1).strip(), km.group(2).strip()
                            if not v or k in ("offenders",):  # skip noisy structural fields
                                continue
                            parts.append(f"{k}={v}")
                        with open(results_file, "a", encoding="utf-8") as f:
                            f.write(f"{tag:55s} epoch={ep:2d}  {' | '.join(parts)}\n")
                            f.flush()
                except:
                    pass
        proc.wait(timeout=TIMEOUT)
        elapsed = time.time() - t0
        ok = proc.returncode == 0 and len(stu_accs) > 0
        best_stu = max(stu_accs.values()) if stu_accs else 0.0
        best_ema = max(ema_accs.values()) if ema_accs else 0.0
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
        elapsed = time.time() - t0
        ok, best_stu, best_ema = False, 0.0, 0.0
    except Exception:
        traceback.print_exc()
        elapsed = time.time() - t0
        ok, best_stu, best_ema = False, 0.0, 0.0
    finally:
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        for _rd in [f"results/BOE_src_only/iter{TARGET_ITER}",
                     f"results/BOE_to_TMI/iter{TARGET_ITER}"]:
            if os.path.isdir(_rd):
                shutil.rmtree(_rd, ignore_errors=True)

    status = "OK" if ok else "FAIL"
    swa_v = swa_acc if swa_acc is not None else 0.0
    stu_all = statistics.mean(stu_accs.values()) if stu_accs else 0.0   # all-round mean
    ema_all = statistics.mean(ema_accs.values()) if ema_accs else 0.0
    print(f"[{done}/{total}] {tag:50s} best_stu={best_stu:.4f} best_ema={best_ema:.4f} swa={swa_v:.4f} time={elapsed:.0f}s {status}")
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(f"{tag:55s} BEST_STU={best_stu:.4f}  BEST_EMA={best_ema:.4f}  SWA_FINAL={swa_v:.4f}  seed={seed}  n_epochs={len(stu_accs)}  time={elapsed:.0f}s  {status}\n")
        if swa_metrics is not None:
            _swa_str = " ".join("{}={:.4f}".format(k, swa_metrics[k]) for k in
                                 ['acc', 'auc', 'recall', 'precision', 'f1', 'bacc', 'specificity', 'kappa', 'gmean', 'mcc'])
            f.write(f"{tag:55s} SWA_METRICS {_swa_str}\n")
        f.write(f"{tag:55s} ALLROUND stu_mean={stu_all:.4f}  ema_mean={ema_all:.4f}\n")
        f.flush()
    return best_stu, best_ema, swa_v, stu_all, ema_all, elapsed, ok


def main():
    start = datetime.now()
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"# batch experiments | {start:%Y-%m-%d %H:%M}\n")
        f.write(f"# configs={len(CONFIGS)} total runs={N_SEEDS_TOTAL}\n")
        f.write("=" * 120 + "\n\n")
    done = 0
    swa_list, stu_all_list, ema_all_list = [], [], []
    for _, (name, overrides, seeds) in enumerate(CONFIGS):
        for seed in seeds:
            done += 1
            bs, be, sw, sa, ea, el, ok = run_one(name, overrides, seed, done, N_SEEDS_TOTAL, RESULTS_FILE)
            if ok:
                swa_list.append(sw)
                stu_all_list.append(sa)
                ema_all_list.append(ea)
    total_elapsed = (datetime.now() - start).total_seconds()
    print(f"\nALL DONE. {N_SEEDS_TOTAL} runs in {total_elapsed/3600:.1f}h")
    if len(swa_list) >= 2:
        print(f"# SWA FINAL mean={statistics.mean(swa_list):.4f} std={statistics.stdev(swa_list):.4f}  (main paper metric, when --swa 1)")
        print(f"# student ALL-round acc mean={statistics.mean(stu_all_list):.4f} std={statistics.stdev(stu_all_list):.4f}")
        print(f"# EMA ALL-round acc     mean={statistics.mean(ema_all_list):.4f} std={statistics.stdev(ema_all_list):.4f}")
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n# done: {N_SEEDS_TOTAL} runs in {total_elapsed/3600:.1f}h\n")
        if len(swa_list) >= 2:
            f.write(f"# SWA FINAL mean={statistics.mean(swa_list):.4f} std={statistics.stdev(swa_list):.4f}\n")
            f.write(f"# student ALL-round acc mean={statistics.mean(stu_all_list):.4f} std={statistics.stdev(stu_all_list):.4f}\n")
            f.write(f"# EMA ALL-round acc     mean={statistics.mean(ema_all_list):.4f} std={statistics.stdev(ema_all_list):.4f}\n")


if __name__ == "__main__":
    main()

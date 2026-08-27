# -*- coding: utf-8 -*-
"""Measure inference time of real trained FCCAN models (paper Sec. 4.10 final).

Per task: 1) train FCCAN with the task's best config (seed=42, iter1);
2) load the saved EMA teacher weights (the main-table eval model);
3) time inference over the whole target test set (batch 16, N repeats) and
report params/FLOPs.

Usage (server):
    python measure_latency_trained.py --tasks "A->B" --seed 42 --repeats 7 --batch 16
Output: measure_latency_trained_results.csv + console summary.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

import torch
import torch.nn as nn

from models.fea_net import FEANet, Classifier

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUT_CSV = os.path.join(HERE, "measure_latency_trained_results.csv")
INPUT_SIZE = 224
NUM_CLASSES = 3
CELL_SPLIT = "CELL_split_2025"

# Task best configs, identical to best/SUMMARY.txt.
BEST_TASKS = [
    {"tag": "A->B", "only": "BOE->TMI", "es": 5, "et": 8,
     "extra": ["--src_ll_prob", "0.7", "--src_ll_alpha", "2.0"]},
    {"tag": "A->C", "only": "BOE->CELL", "es": 4, "et": 15,
     "extra": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.001", "--alpha_scon", "0.01"]},
    {"tag": "B->C", "only": "TMI->CELL", "es": 8, "et": 15,
     "extra": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.5"]},
]


# ---- training ----
def train_one(tag, cfg, seed):
    """Train via main.py in a subprocess; return the save dir."""
    only = cfg["only"]
    src, tgt = only.split("->")
    save_dir = os.path.join("saves", "FEANet_{}_to_{}_iter1".format(src, tgt))
    if not os.path.isfile("main.py"):
        print("[ERROR] main.py not found under {}; run from the project root."
              .format(os.getcwd()))
        sys.exit(1)
    if os.path.isdir(save_dir):
        print("[train] removing old iter1 dir: {}".format(save_dir))
        shutil.rmtree(save_dir, ignore_errors=True)

    cmd = [sys.executable, "main.py", "--only", only,
           "-es", str(cfg["es"]), "-et", str(cfg["et"]),
           "--seed", str(seed), "-i", "1",
           "--cell_split", CELL_SPLIT,
           "--save_ema_weights", "1",
           "--save_which", "ema",
           "--ema_teacher", "0.99",
           "--ema_guide_caco", "1.0",
           "--ema_warmup_epochs", "8",
           "--save_result_txt", "0"] + cfg["extra"]
    print("\n[train] {} seed={}  cmd: {}".format(tag, seed, " ".join(cmd)))
    log_path = os.path.join(HERE, "train_log.txt")
    log_handle = open(log_path, "a", encoding="utf-8")
    log_handle.write("\n\n===== {} seed={} cmd: {} =====\n".format(
        tag, seed, " ".join(cmd)))
    log_handle.flush()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8",
                                errors="replace")
    except Exception as e:
        print("[ERROR] cannot start training: {}".format(e))
        sys.exit(1)
    for line in proc.stdout:
        line = line.rstrip()
        log_handle.write(line + "\n")
        log_handle.flush()
        if ('%|' in line) or ('it/s' in line):
            continue
        if line:
            print(line)
    proc.wait()
    log_handle.close()
    if proc.returncode != 0:
        print("[ERROR] {} training failed returncode={} (see {})"
              .format(tag, proc.returncode, log_path))
        sys.exit(1)
    print("[train] {} done -> {}".format(tag, save_dir))
    # Post-training check: dir existence + content
    if not os.path.isdir(save_dir):
        print("[ERROR] returncode 0 but save dir missing: {}".format(save_dir))
        if os.path.isdir("saves"):
            print("  dirs under saves/: {}".format(os.listdir("saves")))
        else:
            print("  saves/ missing too (training never started? see train_log.txt)")
        sys.exit(1)
    print("[train] saved files: {}".format(sorted(os.listdir(save_dir))))
    return save_dir


# ---- load trained weights ----
def build_fccan_from_weights(save_dir, device):
    """Build FEA-Net + classifier head, load EMA teacher weights (with fallback)."""
    enc = FEANet(wrb_alpha=0.4, wrb_lambda=0.3, use_hf_comp=True, hf_comp_scale=0.2,
                 use_msw_sa=True, msw_sa_positions='3', use_wrb_after_layer2=True)
    clf = Classifier(enc.combined_features, NUM_CLASSES, prob=0.3)

    # Fallback: ema_encoder.pt -> target_encoder.pt.
    candidates = [("ema_encoder.pt", "ema_classifier.pt"),
                  ("target_encoder.pt", "classifier.pt")]
    enc_p = clf_p = loaded = None
    for _e, _c in candidates:
        if os.path.exists(os.path.join(save_dir, _e)) and \
           os.path.exists(os.path.join(save_dir, _c)):
            enc_p = os.path.join(save_dir, _e)
            clf_p = os.path.join(save_dir, _c)
            loaded = _e
            break
    if enc_p is None:
        dir_list = os.listdir(save_dir) if os.path.isdir(save_dir) else ['(dir missing)']
        raise FileNotFoundError(
            "No target weights found. Dir {} contents: {}\n"
            "If only source_encoder.pt/classifier.pt exist, the server main.py is outdated\n"
            "(no target weights saved); sync main.py + trainers/target_trainer.py and rerun."
            .format(save_dir, dir_list))
    print("[load] {}  <- {} / {}".format(save_dir, loaded,
                                          os.path.basename(clf_p)))
    enc.load_state_dict(torch.load(enc_p, map_location="cpu"))
    clf.load_state_dict(torch.load(clf_p, map_location="cpu"))

    class M(nn.Module):
        def __init__(self, enc, clf):
            super().__init__()
            self.enc = enc
            self.clf = clf

        def forward(self, x):
            f, _ = self.enc(x)
            logits, _ = self.clf(f)
            return logits
    model = M(enc, clf).to(device)
    return model


# ---- metrics ----
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model, size=INPUT_SIZE):
    hooks, macs = [], [0]

    def _hook_conv(module, inp, out):
        x = inp[0]
        macs[0] += out.numel() * x.size(1) * module.kernel_size[0] * module.kernel_size[1] // module.groups

    def _hook_linear(module, inp, out):
        macs[0] += module.in_features * out.size(-1)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_hook_conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_hook_linear))
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.randn(1, 3, size, size, device=dev))
    for h in hooks:
        h.remove()
    return macs[0] * 2 / 1e9


def measure_testset_time(model, loader, repeats=7, warmup_batches=2):
    """Time inference over the whole test set (GPU); return (total_s, n_images)."""
    model.eval()
    n_images = len(loader.dataset)
    acc_ms = 0.0
    with torch.no_grad():
        for _ in range(repeats):
            for i, (xs, _) in enumerate(loader):
                if i >= warmup_batches:
                    break
                if torch.cuda.is_available():
                    xs = xs.cuda()
                model(xs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for xs, _ in loader:
                if torch.cuda.is_available():
                    xs = xs.cuda()
                model(xs)
            torch.cuda.synchronize()
            acc_ms += (time.perf_counter() - t0) * 1e3
    return acc_ms / repeats / 1e3, n_images


def main():
    ap = argparse.ArgumentParser(description='Inference time of real trained FCCAN models (3 tasks)')
    ap.add_argument('--tasks', nargs='+', default=[t["tag"] for t in BEST_TASKS],
                    choices=[t["tag"] for t in BEST_TASKS], help='task list')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--repeats', type=int, default=7, help='repeats over the whole test set')
    ap.add_argument('--batch', type=int, default=16, help='inference batch size')
    ap.add_argument('--skip-train', action='store_true',
                    help='skip training; use existing iter1 EMA weights in saves/')
    ap.add_argument('--keep-weights', action='store_true',
                    help='keep weights after measuring (default: delete to save space)')
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("[ERROR] CUDA GPU required (for both training and timing).")
        sys.exit(1)
    dev = torch.device('cuda')
    print("GPU: {}  torch={}  seed={}  repeats={}  batch={}".format(
        torch.cuda.get_device_name(0), torch.__version__, args.seed,
        args.repeats, args.batch))

    from comparison_experiments.common.data_loader import load_task

    rows = []
    for cfg in BEST_TASKS:
        if cfg["tag"] not in args.tasks:
            continue
        tag, only = cfg["tag"], cfg["only"]
        src_name, tgt_name = only.split("->")
        # load_task uses labels A/B/C (BOE/TMI/CELL)
        src_label = {"BOE": "A", "TMI": "B", "CELL": "C"}[src_name]
        tgt_label = {"BOE": "A", "TMI": "B", "CELL": "C"}[tgt_name]
        if not args.skip_train:
            save_dir = train_one(tag, cfg, args.seed)
        else:
            save_dir = os.path.join("saves", "FEANet_{}_to_{}_iter1".format(src_name, tgt_name))
        print("\n[measure] {} loading EMA weights ...".format(tag))
        model = build_fccan_from_weights(save_dir, dev)
        params = count_params(model)
        flops = count_flops(model)

        data = load_task(src_label, tgt_label, input_size=INPUT_SIZE,
                         batch_src=args.batch, batch_tgt=args.batch,
                         tmi_target_unlabeled_pct=50)
        loader = data['tgt_test']
        total_s, n_img = measure_testset_time(model, loader, repeats=args.repeats)
        ms_per = total_s * 1e3 / n_img
        print("[result] {}  n={}  total={:.3f}s  ms/img={:.3f}  params={:.2f}M  flops={:.3f}G"
              .format(tag, n_img, total_s, ms_per, params, flops))
        rows.append({"task": tag, "seed": args.seed, "n_images": n_img,
                     "params_M": round(params, 2), "flops_G": round(flops, 3),
                     "total_s": round(total_s, 3), "ms_per_image": round(ms_per, 3)})
        # Delete the weight dir after measuring to save space.
        if not args.skip_train and not args.keep_weights:
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir, ignore_errors=True)
                print("[clean] deleted weight dir {}".format(save_dir))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "seed", "n_images", "params_M",
                                          "flops_G", "total_s", "ms_per_image"])
        w.writeheader()
        w.writerows(rows)
    print("\nwritten to {}".format(OUT_CSV))


if __name__ == "__main__":
    main()

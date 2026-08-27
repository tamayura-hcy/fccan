#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics proof of the frequency-band modules for 15 models (3 tasks x 5 seeds).

Phase 1 (train): train 3 tasks x 5 seeds with main.py's best config; weights are
copied to <out>/models/{task}_s{seed}/. Phase 2 (analyze): probe MSW-SA / WRB /
HFComp at inference time on each task's target test set; write txt/csv + figures.

Usage (server):
    python run_physics_15.py --phase both     # train + analyze
    python run_physics_15.py --phase train    # train only (resumable)
    python run_physics_15.py --phase analyze  # analyze only
Options: --seeds 42,123,777,2024,3407  --tasks A-B,A-C,B-C  --n_per_class 8  --gradcam 0/1
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

# Set before importing torch to avoid OpenMP runtime conflicts (OMP Error #15).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(8, os.cpu_count() or 4))))
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage, stats as st
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from models.fea_net import FEANet, Classifier
from util.wavelet_recal import haar_dwt2d, haar_idwt2d
import util.data_utils as du

OUT = os.path.join(ROOT, "physics15")
MODEL_DIR = os.path.join(OUT, "models")
LOG_DIR = os.path.join(OUT, "logs")
ANALYSIS_DIR = os.path.join(OUT, "analysis")

SEEDS = [42, 123, 777, 2024, 3407]
CELL_SPLIT = "CELL_split_2025"

# Best config, identical to run_best_metrics.py / best.md.
BEST_TASKS = [
    {"tag": "A-B", "src": "BOE", "tgt": "TMI", "only": "BOE->TMI", "es": 5, "et": 8,
     "extra": ["--src_ll_prob", "0.7", "--src_ll_alpha", "2.0"]},
    {"tag": "A-C", "src": "BOE", "tgt": "CELL", "only": "BOE->CELL", "es": 4, "et": 15,
     "extra": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.001", "--alpha_scon", "0.01"]},
    {"tag": "B-C", "src": "TMI", "tgt": "CELL", "only": "TMI->CELL", "es": 8, "et": 15,
     "extra": ["--lambda_caco", "0.01", "--lambda_batch_ang", "0.5"]},
]

TEST_RE = re.compile(
    r'test acc=(-?[\d.]+) \| auc=(-?[\d.]+) \| recall=(-?[\d.]+) \| precision=(-?[\d.]+) \| f1=(-?[\d.]+)'
    r' \| bacc=(-?[\d.]+) \| specificity=(-?[\d.]+) \| kappa=(-?[\d.]+) \| gmean=(-?[\d.]+) \| mcc=(-?[\d.]+)')

CAND_FILES = ["ema_encoder.pt", "ema_classifier.pt", "target_encoder.pt", "classifier.pt",
              "source_encoder.pt", "source_encoder_best.pt", "classifier_best.pt"]


DRY_RUN = False


# ---------- Phase 1: train ----------
def run_one_training(cfg, seed, log_handle, force_retrain=False):
    tag, only = cfg["tag"], cfg["only"]
    src, tgt = cfg["src"], cfg["tgt"]
    model_dir = os.path.join(MODEL_DIR, "{}_s{}".format(tag, seed))
    ema_p = os.path.join(model_dir, "ema_encoder.pt")
    tgt_p = os.path.join(model_dir, "target_encoder.pt")
    if (os.path.exists(ema_p) or os.path.exists(tgt_p)) and not force_retrain:
        print("  [SKIP] {} s{} model exists, skipping".format(tag, seed))
        return None
    if force_retrain:
        shutil.rmtree(model_dir, ignore_errors=True)

    cmd = [sys.executable, "main.py", "--only", only,
           "-es", str(cfg["es"]), "-et", str(cfg["et"]),
           "--seed", str(seed), "-i", "1",
           "--cell_split", CELL_SPLIT,
           "--save_result_txt", "1",
           "--save_ema_weights", "1"] + cfg["extra"]
    print("\n[RUN] {} s{}  only={} es={} et={}".format(tag, seed, only, cfg["es"], cfg["et"]))
    if DRY_RUN:
        print("    [DRY] {}".format(" ".join(cmd)))
        return None
    if log_handle:
        log_handle.write("\n[RUN] {} s{}  only={} es={} et={}\n".format(tag, seed, only, cfg["es"], cfg["et"]))
        log_handle.flush()

    t0 = time.time()
    last_m = None
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.rstrip()
            if ("%|" in line) or ("it/s" in line):
                continue
            if line:
                print(line)
            if log_handle:
                log_handle.write(line + "\n")
                log_handle.flush()
            _m = TEST_RE.search(line)
            if _m:
                last_m = {k: float(v) for k, v in zip(
                    ["acc", "auc", "recall", "precision", "f1", "bacc", "specificity",
                     "kappa", "gmean", "mcc"], _m.groups())}
        proc.wait()
    except Exception as e:
        print("[ERROR] {} s{}: {}".format(tag, seed, e))
        return None
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()

    if proc.returncode != 0:
        print("[FAIL] {} s{} exit={}".format(tag, seed, proc.returncode))
        return None

    # Copy weights right away (the source save dir is shared across seeds).
    save_name = "./saves/FEANet_{}_to_{}_iter1".format(src, tgt)
    os.makedirs(model_dir, exist_ok=True)
    for f in CAND_FILES:
        p = os.path.join(save_name, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(model_dir, f))
    with open(os.path.join(model_dir, "final_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("task={} seed={} done at {}\n".format(tag, seed, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        if last_m:
            for k, v in last_m.items():
                f.write("  {:<12} {:.4f}\n".format(k, v))
    msg = "[OK] {} s{} done ({:.0f}s) last acc = {}".format(
        tag, seed, time.time() - t0, last_m["acc"] if last_m else "n/a")
    print(msg)
    if log_handle:
        log_handle.write(msg + "\n")
        log_handle.flush()
    return last_m


def phase_train(args, seeds, tasks):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    log_handle = open(os.path.join(LOG_DIR, "train.log"), "a", encoding="utf-8")
    print("Phase 1 train: {} tasks x {} seeds = {} runs".format(len(tasks), len(seeds), len(tasks) * len(seeds)))
    results = {}
    for cfg in tasks:
        results[cfg["tag"]] = {}
        for seed in seeds:
            m = run_one_training(cfg, seed, log_handle, force_retrain=args.force_retrain)
            if m:
                results[cfg["tag"]][seed] = m
    log_handle.close()
    # Training summary
    path = os.path.join(OUT, "train_summary.txt")
    lines = ["3 tasks x 5 seeds training summary (EMA target test)\n"]
    for cfg in tasks:
        tag = cfg["tag"]
        lines.append("\n[{}] {}->{}".format(tag, cfg["src"], cfg["tgt"]))
        for seed in seeds:
            m = results[tag].get(seed)
            if m:
                lines.append("  s{:<6} acc={:.4f} auc={:.4f}".format(seed, m["acc"], m["auc"]))
            else:
                lines.append("  s{:<6} FAILED/MISSING".format(seed))
        vals = [results[tag][s]["acc"] for s in seeds if s in results[tag]]
        if vals:
            lines.append("  mean acc = {:.4f} +/- {:.4f} ({} seeds)".format(
                sum(vals) / len(vals), (sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / len(vals)) ** 0.5,
                len(vals)))
    txt = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


# ---------- Phase 2: physics analysis ----------
def band_rms_map(x):
    """x (1,C,H,W) -> dict of per-band RMS using the correct per-channel subband view."""
    sub = haar_dwt2d(x)
    per_ch = subbands_per_channel(sub)          # (B,C,4,h,w)
    out = {}
    for k, key in enumerate(("ll", "lh", "hl", "hh")):
        out[key] = float(per_ch[:, :, k].pow(2).mean().sqrt())
    return out


def subbands_per_channel(sub):
    """Rearrange haar_dwt2d's (B,4,C,h,w) view to (B,C,4,h,w) (output channel g*4+k)."""
    B, K, C, h, w = sub.shape
    out = torch.empty(B, C, K, h, w, device=sub.device, dtype=sub.dtype)
    for g in range(C):
        for k in range(K):
            c = g * K + k
            kp, gp = divmod(c, C)
            out[:, g, k] = sub[:, kp, gp]
    return out


def idwt_correct(sub_per_ch):
    """Correct Haar iDWT, exact inverse of haar_dwt2d: (B,C,4,h,w) -> (B,C,2h,2w)."""
    B, C, K, h, w = sub_per_ch.shape
    LL, LH, HL, HH = [sub_per_ch[:, :, k] for k in range(4)]
    out = torch.zeros(B, C, 2 * h, 2 * w, device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = 0.5 * (LL + LH + HL + HH)
    out[:, :, 0::2, 1::2] = 0.5 * (LL - LH + HL - HH)
    out[:, :, 1::2, 0::2] = 0.5 * (LL + LH - HL - HH)
    out[:, :, 1::2, 1::2] = 0.5 * (LL - LH - HL + HH)
    return out


def att_from_pooled(pooled, ms):
    """Replicate WaveletSpatialAttentionLite.forward's attention computation."""
    H, W = pooled.size(2), pooled.size(3)
    sub = haar_dwt2d(pooled)
    LL = sub[:, 0].contiguous()
    a = ms.conv_ll1(LL)
    a = ms.bn_ll1(a)
    a = F.relu(a, inplace=True)
    a = ms.conv_ll2(a)
    a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
    return torch.sigmoid(a), sub


def slab_mask(gray):
    prof = gray.mean(axis=1)
    on = prof > 0.25 * prof.max()
    best = (0, 0)
    cur = None
    for i, v in enumerate(on):
        if v and cur is None:
            cur = i
        if (not v or i == len(on) - 1) and cur is not None:
            end = i if not v else i + 1
            if end - cur > best[1] - best[0]:
                best = (cur, end)
            cur = None
    m = np.zeros_like(gray, dtype=bool)
    m[best[0]:best[1], :] = True
    return m


def cam_map(A, logits, cls, size=224):
    A.retain_grad()
    logits[0, cls].backward()
    grad = A.grad[0]
    alpha = grad.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((alpha * A[0]).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(size, size),
                        mode="bilinear", align_corners=False)[0, 0]
    return cam.detach().cpu().numpy()


def cam_in_ratio(cam, mask):
    c = np.clip(cam.astype(np.float64), 0, None)
    if c.sum() <= 0:
        return np.nan
    return float((c / c.sum())[mask].sum())


# ---- Module probes (replicate module forwards, return intermediate tensors) ----
def wrb_probe(wrb, x):
    """Replicate WaveletRecalibrationBlock.forward, return subbands/gates/scale/lam."""
    sub = haar_dwt2d(x)
    sub[:, 3] = 0.0
    B, K, C, h, w = sub.shape
    g = wrb.gate(sub.reshape(B, -1, h, w))            # (B,4)
    scale = 1.0 + wrb.alpha * torch.tanh(g)           # (B,4)
    sub_s = sub * scale.view(B, 4, 1, 1, 1)
    x_rec = haar_idwt2d(sub_s)
    lam = torch.sigmoid(wrb.lambda_param) * 0.5 + 0.25
    out = x + lam * wrb.conv1x1(x_rec)
    return out, sub, sub_s, scale, lam, x_rec


def hf_probe(hf, x):
    """Replicate HFCompBlock.forward, return residual and SE weights."""
    sub = haar_dwt2d(x)
    LL, LH, HL, HH = sub[:, 0], sub[:, 1], sub[:, 2], sub[:, 3]
    hf_in = LH + HL
    y = hf.conv(hf_in)
    w = hf.se(y)
    y = y * w
    y_each = y / 2.0
    sub_hf = torch.stack([torch.zeros_like(LL), y_each, y_each, torch.zeros_like(HH)], dim=1)
    x_hf = haar_idwt2d(sub_hf)
    return hf.scale * x_hf, w, sub, y


def forward_manual(c, clf, x, msw_on=True, wrb_mode="on", hf_on=True):
    """Manual forward matching FEAWithWRB.forward; returns intermediates.

    wrb_mode: 'on' = actual WRB; 'off' = skip WRB (y=x); 'correct' = correct-iDWT WRB.
    Returns: h0 (layer3 out before MSW-SA), att/sub (MSW-SA gate + Haar coeffs),
    h_msw, h_wrb_in/out, h_wrb_corr, wrb_early_info, wrb1024_info, h_hf_in/res,
    se_w, sub_hf, h_final, A, logits.
    """
    c = getattr(c, "cnn", c) if hasattr(c, "cnn") and not hasattr(c, "conv1") else c
    h = c.conv1(x)
    h = c.bn1(h)
    h = c.relu(h)
    h = c.maxpool(h)
    h = c.layer1(h)
    h = c.layer2(h)
    wrb_early_info = None
    if c.wrb_early is not None:
        if wrb_mode == "on":
            h, sub_e, sub_s_e, scale_e, lam_e, x_rec_e = wrb_probe(c.wrb_early, h)
            wrb_early_info = dict(scale=scale_e, lam=lam_e, sub=sub_e, sub_s=sub_s_e)
        elif wrb_mode == "correct":
            sub_e = haar_dwt2d(h)
            sub_e[:, 3] = 0.0
            B, K, Cc, hh, ww = sub_e.shape
            g = c.wrb_early.gate(sub_e.reshape(B, -1, hh, ww))
            sc = 1.0 + c.wrb_early.alpha * torch.tanh(g)
            sub_s = sub_e * sc.view(B, 4, 1, 1, 1)
            x_rec = idwt_correct(subbands_per_channel(sub_s))
            lam = torch.sigmoid(c.wrb_early.lambda_param) * 0.5 + 0.25
            h = h + lam * c.wrb_early.conv1x1(x_rec)
            wrb_early_info = dict(scale=sc, lam=lam)
        # 'off': keep h unchanged
    if c.msw_sa2 is not None and msw_on:
        h = c.msw_sa2(h)
    h0 = c.layer3(h)
    if msw_on and c.msw_sa3 is not None:
        mx, _ = h0.max(1, keepdim=True)
        av = h0.mean(1, keepdim=True)
        pooled = torch.cat([mx, av], 1)
        att, sub = att_from_pooled(pooled, c.msw_sa3)
        h_msw = h0 + h0 * att
    else:
        att, sub, pooled = None, None, None
        h_msw = h0

    h_wrb_in = h_msw
    wrb1024_info = None
    if wrb_mode == "on":
        h_wrb_out, sub_w, sub_s_w, scale_w, lam_w, x_rec_w = wrb_probe(c.wrb, h_wrb_in)
        # Correct-iDWT reference
        sub_sw = sub_s_w
        x_rec_c = idwt_correct(subbands_per_channel(sub_sw))
        h_wrb_corr = h_wrb_in + lam_w * c.wrb.conv1x1(x_rec_c)
        wrb1024_info = dict(scale=scale_w, lam=lam_w, sub=sub_w, sub_s=sub_s_w, x_rec=x_rec_w)
    elif wrb_mode == "correct":
        sub_w = haar_dwt2d(h_wrb_in)
        sub_w[:, 3] = 0.0
        B, K, Cc, hh, ww = sub_w.shape
        g = c.wrb.gate(sub_w.reshape(B, -1, hh, ww))
        sc = 1.0 + c.wrb.alpha * torch.tanh(g)
        sub_sw = sub_w * sc.view(B, 4, 1, 1, 1)
        x_rec = idwt_correct(subbands_per_channel(sub_sw))
        lam = torch.sigmoid(c.wrb.lambda_param) * 0.5 + 0.25
        h_wrb_out = h_wrb_in + lam * c.wrb.conv1x1(x_rec)
        h_wrb_corr = h_wrb_out
        wrb1024_info = dict(scale=sc, lam=lam, sub=sub_w, sub_s=sub_sw)
    else:  # off
        h_wrb_out = h_wrb_in
        h_wrb_corr = h_wrb_in

    h_hf_in = h_wrb_out + c.wt_conv_branch(h_wrb_out)
    if hf_on and c.hf_comp is not None:
        h_hf_res, se_w, sub_hf, y_hf = hf_probe(c.hf_comp, h_hf_in)
    else:
        h_hf_res = torch.zeros_like(h_hf_in)
        se_w, sub_hf, y_hf = None, None, None
    h_final = h_hf_in + h_hf_res

    A = c.layer4(h_final)
    g = c.avgpool(A).flatten(1)
    logits, _ = clf(g)
    return dict(h0=h0, att=att, sub=sub, pooled=pooled, h_msw=h_msw,
                h_wrb_in=h_wrb_in, h_wrb_out=h_wrb_out, h_wrb_corr=h_wrb_corr,
                wrb_early_info=wrb_early_info, wrb1024_info=wrb1024_info,
                h_hf_in=h_hf_in, h_hf_res=h_hf_res, se_w=se_w, sub_hf=sub_hf,
                h_final=h_final, A=A, logits=logits)


def margin_of(logits):
    lo = logits[0].detach().cpu().numpy()
    s = np.sort(lo)
    return float(s[-1] - s[-2])


def analyze_sample(enc, clf, x, idx, cls_name, name, do_gradcam=True, do_erf=False):
    """Measure all three modules' physics on one sample; returns a row dict."""
    dev = x.device
    c = enc.cnn
    row = dict(task=None, seed=None, cls=cls_name, name=name, idx=idx)

    # ---- Main forward (all modules on) ----
    res_on = forward_manual(c, clf, x, msw_on=True, wrb_mode="on", hf_on=True)
    row["pred"] = int(res_on["logits"][0].argmax())
    row["m_on"] = margin_of(res_on["logits"])
    row["m_off_msw"] = margin_of(forward_manual(c, clf, x, msw_on=False, wrb_mode="on", hf_on=True)["logits"])
    row["m_off_wrb"] = margin_of(forward_manual(c, clf, x, msw_on=True, wrb_mode="off", hf_on=True)["logits"])
    row["m_off_hf"] = margin_of(forward_manual(c, clf, x, msw_on=True, wrb_mode="on", hf_on=False)["logits"])

    with torch.no_grad():
        # ---------- MSW-SA ----------
        att = res_on["att"]
        sub = res_on["sub"]
        ms = c.msw_sa3
        if ms is not None and att is not None:
            s14 = att[0, 0].detach().cpu().numpy()
            # A1 exact invariance
            per_ch = subbands_per_channel(sub)               # (1,2,4,7,7)
            sub_kept = torch.zeros_like(sub)
            sub_kept[:, 0] = sub[:, 0]
            pooled_kept = idwt_correct(subbands_per_channel(sub_kept))
            S_kept, _ = att_from_pooled(pooled_kept, ms)
            row["d_kept"] = float((att - S_kept).abs().max())
            sub_none = torch.zeros_like(sub)
            sub_none[:, 1:] = sub[:, 1:]
            pooled_none = idwt_correct(subbands_per_channel(sub_none))
            S_none, _ = att_from_pooled(pooled_none, ms)
            row["d_none"] = float((att - S_none).abs().max())
            row["std_none"] = float(S_none[0, 0].std())
            coef_intended = torch.stack([sub[:, 0, 0], sub[:, 2, 0]], dim=1).contiguous()
            a2 = ms.conv_ll1(coef_intended)
            a2 = ms.bn_ll1(a2)
            a2 = F.relu(a2, inplace=True)
            a2 = ms.conv_ll2(a2)
            S_int = torch.sigmoid(F.interpolate(a2, size=(14, 14), mode="bilinear",
                                                align_corners=False))
            row["d_intended"] = float((att - S_int).abs().max())
            # A3 modulation range
            mod = 1.0 + s14
            row["mod_min"] = float(mod.min())
            row["mod_max"] = float(mod.max())
            # A4 alignment
            g_img = x[0, 0].detach().cpu().numpy()
            mask = slab_mask(g_img)
            m14 = ndimage.zoom(mask.astype(np.float32), (14 / 224, 14 / 224), order=0) > 0.5
            subi = haar_dwt2d(torch.from_numpy(g_img).float().unsqueeze(0).unsqueeze(0))
            ll14 = ndimage.zoom(subi[0, 0, 0].numpy(), (14 / 112, 14 / 112), order=1)
            hh14 = ndimage.zoom(subi[0, 3, 0].numpy(), (14 / 112, 14 / 112), order=1)
            row["r_ll"] = st.pearsonr(s14.ravel(), ll14.ravel()).statistic
            row["r_hh"] = st.pearsonr(s14.ravel(), hh14.ravel()).statistic
            row["s_in"] = float(s14[m14].mean()) if m14.any() else np.nan
            row["s_out"] = float(s14[~m14].mean()) if (~m14).any() else np.nan
        else:
            for k in ("d_kept", "d_none", "std_none", "d_intended", "mod_min", "mod_max",
                      "r_ll", "r_hh", "s_in", "s_out"):
                row[k] = np.nan

        # ---------- WRB (1024 main + early 512 gate) ----------
        wi = res_on["wrb1024_info"]
        if wi is not None:
            # Gate scale
            sc = wi["scale"][0].cpu().numpy()
            row["w_g_ll"], row["w_g_lh"], row["w_g_hl"], row["w_g_hh"] = map(float, sc)
            # Band RMS in/out
            rms_in = band_rms_map(res_on["h_wrb_in"])
            rms_out = band_rms_map(res_on["h_wrb_out"])
            for k in ("ll", "lh", "hl", "hh"):
                row["w_rms_in_" + k] = rms_in[k]
                row["w_rms_out_" + k] = rms_out[k]
            # Residual band composition
            res = res_on["h_wrb_out"] - res_on["h_wrb_in"]
            rms_res = band_rms_map(res)
            for k in ("ll", "lh", "hl", "hh"):
                row["w_res_" + k] = rms_res[k]
            # Implementation deviation (actual vs correct iDWT)
            row["w_d_rec"] = float((res_on["h_wrb_out"] - res_on["h_wrb_corr"]).abs().max())
        ei = res_on["wrb_early_info"]
        if ei is not None:
            sce = ei["scale"][0].cpu().numpy()
            row["e_g_ll"], row["e_g_lh"], row["e_g_hl"], row["e_g_hh"] = map(float, sce)

        # ---------- HFComp ----------
        if res_on["h_hf_res"] is not None and c.hf_comp is not None:
            hres = res_on["h_hf_res"]
            rms_res = band_rms_map(hres)
            for k in ("ll", "lh", "hl", "hh"):
                row["h_res_" + k] = rms_res[k]
            rms_in = band_rms_map(res_on["h_hf_in"])
            rms_out = band_rms_map(res_on["h_final"])
            for k in ("ll", "lh", "hl", "hh"):
                row["h_rms_in_" + k] = rms_in[k]
                row["h_rms_out_" + k] = rms_out[k]
            sw = res_on["se_w"][0].detach().cpu().numpy()
            row["h_se_mean"] = float(sw.mean())
            row["h_se_std"] = float(sw.std())

    # ---------- Grad-CAM (needs grad, outside no_grad) ----------
    if do_gradcam and ms is not None and att is not None:
        with torch.enable_grad():
            xg = x.detach().clone().requires_grad_(True)
            resg = forward_manual(c, clf, xg, msw_on=True, wrb_mode="on", hf_on=True)
            cam_on = cam_map(resg["A"], resg["logits"], int(resg["logits"][0].argmax()))
            g_img = xg[0, 0].detach().cpu().numpy()
            mask = slab_mask(g_img)
            row["cam_in_on"] = cam_in_ratio(cam_on, mask)
            xg2 = x.detach().clone().requires_grad_(True)
            resg2 = forward_manual(c, clf, xg2, msw_on=False, wrb_mode="on", hf_on=True)
            cam_off = cam_map(resg2["A"], resg2["logits"], int(resg2["logits"][0].argmax()))
            row["cam_in_off"] = cam_in_ratio(cam_off, mask)
    return row


def phase_analyze(args, seeds, tasks):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Phase 2 physics analysis on", dev)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    n_per = int(args.n_per_class)
    do_gradcam = int(args.gradcam) == 1

    all_rows = {}
    for cfg in tasks:
        tag, src, tgt = cfg["tag"], cfg["src"], cfg["tgt"]
        t_data_dir = du.get_data_dir(tgt, use_bg_removed=True, use_baseline_paths=False)
        test_dir = os.path.join(t_data_dir, "test")
        ds = ImageFolder(test_dir, transform=tf)
        classes = ds.classes
        # Evenly sample n_per per class (filename order, fixed across seeds).
        sample_files = {}
        for k, cls in enumerate(classes):
            names = sorted(p for p, l in ds.samples if l == k)
            n = len(names)
            if n == 0:
                continue
            idxs = np.linspace(0, n - 1, n_per, dtype=int)
            sample_files[cls] = [names[i] for i in idxs]
        print("[{}] {} test: {} classes x {} samples per class = {}".format(
            tag, tgt, len(classes), n_per, sum(len(v) for v in sample_files.values())))

        rows = []
        for seed in seeds:
            # Candidate dirs: physics15 copy first, then main.py's save dir.
            cand_dirs = [os.path.join(MODEL_DIR, "{}_s{}".format(tag, seed)),
                         os.path.join("./saves", "FEANet_{}_to_{}_iter1".format(src, tgt))]
            enc_path = clf_path = None
            which = None
            for mdir in cand_dirs:
                for enc_n, clf_n, tag_w in (("ema_encoder.pt", "ema_classifier.pt", "EMA"),
                                            ("target_encoder.pt", "classifier.pt", "STUDENT"),
                                            ("source_encoder.pt", "classifier.pt", "SOURCE(no-UDA!)")):
                    e = os.path.join(mdir, enc_n)
                    c = os.path.join(mdir, clf_n)
                    if os.path.exists(e) and os.path.exists(c):
                        enc_path, clf_path, which = e, c, tag_w
                        break
                if enc_path:
                    break
            if not enc_path:
                print("  [WARN] {} s{} model missing, skipping".format(tag, seed))
                continue
            enc = FEANet(use_hf_comp=True, use_msw_sa=True, msw_sa_positions="3",
                         use_wrb_after_layer2=True)
            enc.load_state_dict(torch.load(enc_path, map_location="cpu"), strict=False)
            clf = Classifier(2048, len(classes), prob=0.0)
            clf.load_state_dict(torch.load(clf_path, map_location="cpu"), strict=False)
            enc.eval().to(dev)
            clf.eval().to(dev)
            print("  [{}] s{} loaded {} ({})".format(tag, seed, os.path.basename(enc_path), which))
            for cls_name, names in sample_files.items():
                for name in names:
                    pil = Image.open(name).convert("RGB")
                    x = tf(pil).unsqueeze(0).to(dev)
                    row = analyze_sample(enc, clf, x, 0, cls_name, os.path.basename(name),
                                         do_gradcam=do_gradcam)
                    row["task"] = tag
                    row["seed"] = seed
                    rows.append(row)
        all_rows[tag] = rows

    # ---------- Summary ----------
    summary_lines = []
    csv_dir = os.path.join(ANALYSIS_DIR, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    for cfg in tasks:
        tag = cfg["tag"]
        rows = all_rows.get(tag, [])
        if not rows:
            continue
        keys = sorted(rows[0].keys())
        import csv
        with open(os.path.join(csv_dir, "{}_rows.csv".format(tag)), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        report = build_report(tag, rows, args)
        summary_lines.extend(report)
        print("\n".join(report))

    txt = "\n".join(summary_lines) + "\n"
    with open(os.path.join(OUT, "summary_physics.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    if not int(args.no_fig):
        try:
            make_figs(all_rows, tasks, ANALYSIS_DIR)
        except Exception as e:
            print("[fig WARN]", e)
    print("\n[done] ->", OUT)


def _summ(vals):
    a = np.array([float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if a.size == 0:
        return "n/a"
    return "{:.4f}±{:.4f}".format(a.mean(), a.std())


def _pairstat(a, b, alt="greater"):
    a = np.array([float(v) for v in a])
    b = np.array([float(v) for v in b])
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or np.allclose(a, b):
        return "n/a", "n/a"
    try:
        p_w = st.wilcoxon(a, b, alternative=alt).pvalue
    except Exception:
        p_w = np.nan
    try:
        p_t = st.ttest_rel(a, b, alternative=alt).pvalue
    except Exception:
        p_t = np.nan
    return "{:.3g}".format(p_w), "{:.3g}".format(p_t)


def build_report(tag, rows, args):
    import numpy as np
    import scipy.stats as st
    L = []
    L.append("=" * 88)
    L.append("PHYSICS REPORT  task={}   models={}  samples/model={}".format(
        tag, len(set(r["seed"] for r in rows)), len(rows) // max(1, len(set(r["seed"] for r in rows)))))
    L.append("=" * 88)
    L.append("[Method note] band energy/RMS use the correct per-channel view subbands_per_channel(haar_dwt2d(x)):")
    L.append("  haar_dwt2d's raw view sub[:,k,g] = conv-output channel k*C+g; the true band k of channel g is at g*4+k;")
    L.append("  taking sub[:,k] directly is NOT band energy (WRB/HFComp internally operate on the raw view).")
    L.append("[Method note] haar_idwt2d is a non-exact inverse of haar_dwt2d (transposed-conv difference); deviations")
    L.append("  are reported as w_d_rec (WRB) and d_intended (MSW-SA design gap);")
    L.append("  MSW-SA only uses the decomposition (C=2), so A1 invariance is unaffected. Physics is based on measured bands.")
    L.append("=" * 88)

    def col(k):
        return [r.get(k) for r in rows]

    # ---------- MSW-SA ----------
    m_on, m_off = col("m_on"), col("m_off_msw")
    p_w, p_t = _pairstat(m_on, m_off)
    L.append("\n[MSW-SA] A1 exact coarse-scale compression (only 2 of 16 Haar coefficient channels used)")
    L.append("   max|S - S(keep-used-2ch)|      = {:.3e}".format(np.nanmax(col("d_kept"))))
    L.append("   max|S - S(drop-used-2ch)|      = {:.3f}   (S nearly constant after removal, std_mean={:.4f})".format(np.nanmax(col("d_none")), np.nanmean(col("std_none"))))
    L.append("   mean|S_actual - S_intended|    = {:.4f}   (impl vs design)".format(np.nanmean(col("d_intended"))))
    L.append("[MSW-SA] A3 data-adaptive gain and decision margin")
    L.append("   modulation 1+S range: {:.3f}~{:.3f} (per-sample extreme means)".format(np.nanmean(col("mod_min")), np.nanmean(col("mod_max"))))
    L.append("   decision margin: ON {:.2f}±{:.2f}  OFF {:.2f}±{:.2f}  (Wilcoxon p={}, t p={})".format(
        np.nanmean(m_on), np.nanstd(m_on), np.nanmean(m_off), np.nanstd(m_off), p_w, p_t))
    r_ll, r_hh = col("r_ll"), col("r_hh")
    p_r, _ = _pairstat(r_ll, r_hh)
    L.append("[MSW-SA] A4 low-frequency structure alignment")
    L.append("   corr(S, LL) = {:.3f}±{:.3f}  vs corr(S, HH) = {:.3f}±{:.3f}  (p={})".format(
        np.nanmean(r_ll), np.nanstd(r_ll), np.nanmean(r_hh), np.nanstd(r_hh), p_r))
    s_in, s_out = col("s_in"), col("s_out")
    p_s, _ = _pairstat(s_in, s_out)
    L.append("   S in-slab = {:.3f}±{:.3f}  vs out-slab = {:.3f}±{:.3f}  (p={})".format(
        np.nanmean(s_in), np.nanstd(s_in), np.nanmean(s_out), np.nanstd(s_out), p_s))
    if "cam_in_on" in rows[0]:
        L.append("   Grad-CAM in-slab ratio: ON {:.3f}±{:.3f} / OFF {:.3f}±{:.3f}".format(
            np.nanmean(col("cam_in_on")), np.nanstd(col("cam_in_on")),
            np.nanmean(col("cam_in_off")), np.nanstd(col("cam_in_off"))))
    p_w2, p_t2 = _pairstat(m_on, col("m_off_wrb"))
    p_w3, p_t3 = _pairstat(m_on, col("m_off_hf"))
    L.append("\n[WRB]   decision margin: ON {:.2f}±{:.2f}  WRB-OFF {:.2f}±{:.2f}  (Wilcoxon p={}, t p={})".format(
        np.nanmean(m_on), np.nanstd(m_on), np.nanmean(col("m_off_wrb")), np.nanstd(col("m_off_wrb")), p_w2, p_t2))
    L.append("[HFComp] decision margin: ON {:.2f}±{:.2f}  HF-OFF {:.2f}±{:.2f}  (Wilcoxon p={}, t p={})".format(
        np.nanmean(m_on), np.nanstd(m_on), np.nanmean(col("m_off_hf")), np.nanstd(col("m_off_hf")), p_w3, p_t3))

    L.append("\n[WRB-1024] gate -> scale=1+0.4*tanh(g), 4 gate groups (the 4 coefficient blocks of the impl view)")
    for k, nm in (("w_g_ll", "g0"), ("w_g_lh", "g1"), ("w_g_hl", "g2"), ("w_g_hh", "g3")):
        L.append("   {}: {:.3f}±{:.3f}".format(nm, np.nanmean(col(k)), np.nanstd(col(k))))
    L.append("[WRB-1024] band RMS out/in ratio (correct freq view; >1 enhance, <1 suppress)")
    for k in ("ll", "lh", "hl", "hh"):
        outs = [o / i if i > 1e-12 else np.nan for o, i in zip(col("w_rms_out_" + k), col("w_rms_in_" + k))]
        L.append("   {}: {:.3f}±{:.3f}".format(k.upper(), np.nanmean(outs), np.nanstd(outs)))
    L.append("[WRB-1024] residual energy ratio res/in (res=out-in, correct freq view)")
    for k in ("ll", "lh", "hl", "hh"):
        rats = [o / i if i > 1e-12 else np.nan for o, i in zip(col("w_res_" + k), col("w_rms_in_" + k))]
        L.append("   {}: {:.4f}±{:.4f}".format(k.upper(), np.nanmean(rats), np.nanstd(rats)))
    L.append("   WRB impl deviation max|out_actual - out_correctIWT| = {:.2e} (non-exact inverse of haar_idwt2d; "
             "physics based on measured bands)".format(np.nanmax(col("w_d_rec"))))
    if "e_g_ll" in rows[0]:
        L.append("[WRB-512 early] gates: g0 {:.3f}±{:.3f}  g1 {:.3f}±{:.3f}  g2 {:.3f}±{:.3f}  g3 {:.3f}±{:.3f}".format(
            np.nanmean(col("e_g_ll")), np.nanstd(col("e_g_ll")),
            np.nanmean(col("e_g_lh")), np.nanstd(col("e_g_lh")),
            np.nanmean(col("e_g_hl")), np.nanstd(col("e_g_hl")),
            np.nanmean(col("e_g_hh")), np.nanstd(col("e_g_hh"))))

    L.append("\n[HFComp] residual band composition (correct freq view; LL/HH ≈ 0, LH/HL dominant)")
    for k in ("ll", "lh", "hl", "hh"):
        L.append("   {}: {:.4f}±{:.4f}".format(k.upper(), np.nanmean(col("h_res_" + k)), np.nanstd(col("h_res_" + k))))
    L.append("[HFComp] compensation res/in (residual vs input subband RMS; LH/HL > 0, LL/HH ≈ 0)")
    for k in ("ll", "lh", "hl", "hh"):
        rats = [o / i if i > 1e-12 else np.nan for o, i in zip(col("h_res_" + k), col("h_rms_in_" + k))]
        L.append("   {}: {:.4f}±{:.4f}".format(k.upper(), np.nanmean(rats), np.nanstd(rats)))
    L.append("[HFComp] out/in band RMS ratio")
    for k in ("ll", "lh", "hl", "hh"):
        outs = [o / i if i > 1e-12 else np.nan for o, i in zip(col("h_rms_out_" + k), col("h_rms_in_" + k))]
        L.append("   {}: {:.3f}±{:.3f}".format(k.upper(), np.nanmean(outs), np.nanstd(outs)))
    L.append("[HFComp] SE channel weights: mean={:.3f}±{:.3f}  per-sample std={:.3f}±{:.3f} (per-sample adaptive)".format(
        np.nanmean(col("h_se_mean")), np.nanstd(col("h_se_mean")),
        np.nanmean(col("h_se_std")), np.nanstd(col("h_se_std"))))
    return L


def make_figs(all_rows, tasks, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    for cfg in tasks:
        tag = cfg["tag"]
        rows = all_rows.get(tag, [])
        if not rows:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=110)
        def col(k):
            return np.array([r.get(k) for r in rows], dtype=float)
        # (0,0) margin of the three groups
        ax = axes[0, 0]
        cats = [("MSW-SA ON", col("m_on")), ("MSW-OFF", col("m_off_msw")),
                ("WRB-OFF", col("m_off_wrb")), ("HF-OFF", col("m_off_hf"))]
        xs = np.arange(len(cats))
        ax.bar(xs, [np.nanmean(v) for _, v in cats], yerr=[np.nanstd(v) for _, v in cats],
               color=["#4472C4", "#C00000", "#C00000", "#C00000"], alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels([n for n, _ in cats], fontsize=8)
        ax.set_title("decision margin (top1-top2 logit)", fontsize=10)
        # (0,1) r_ll vs r_hh
        ax = axes[0, 1]
        ax.boxplot([col("r_ll"), col("r_hh")], labels=["S~LL", "S~HH"], showmeans=True)
        ax.set_title("corr(S, subband)", fontsize=10)
        # (0,2) s_in vs s_out
        ax = axes[0, 2]
        ax.boxplot([col("s_in"), col("s_out")], labels=["slab in", "slab out"], showmeans=True)
        ax.set_title("mean S(F)", fontsize=10)
        # (1,0) WRB gate scale (4 groups)
        ax = axes[1, 0]
        ax.boxplot([col("w_g_ll"), col("w_g_lh"), col("w_g_hl"), col("w_g_hh")],
                   labels=["g0", "g1", "g2", "g3"], showmeans=True)
        ax.axhline(1.0, color="gray", ls=":")
        ax.set_title("WRB gate scale (1+0.4 tanh g)", fontsize=10)
        # (1,1) HF out/in ratio
        ax = axes[1, 1]
        rats = {}
        for k in ("ll", "lh", "hl", "hh"):
            o = col("h_rms_out_" + k)
            i = col("h_rms_in_" + k)
            rats[k] = np.array([oo / ii if ii > 1e-12 else np.nan for oo, ii in zip(o, i)])
        ax.bar(np.arange(4), [np.nanmean(rats[k]) for k in ("ll", "lh", "hl", "hh")],
               yerr=[np.nanstd(rats[k]) for k in ("ll", "lh", "hl", "hh")],
               color=["#4472C4", "#ED7D31", "#ED7D31", "#4472C4"], alpha=0.85)
        ax.axhline(1.0, color="gray", ls=":")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["LL", "LH", "HL", "HH"], fontsize=8)
        ax.set_title("HFComp out/in subband RMS", fontsize=10)
        # (1,2) res composition
        ax = axes[1, 2]
        wr = np.array([np.nanmean(col("w_res_" + k)) for k in ("lh", "hl", "ll", "hh")])
        hr = np.array([np.nanmean(col("h_res_" + k)) for k in ("lh", "hl", "ll", "hh")])
        if np.nansum(wr) > 0:
            wr = wr / np.nansum(wr)
        if np.nansum(hr) > 0:
            hr = hr / np.nansum(hr)
        xw = np.arange(4) - 0.2
        xh = np.arange(4) + 0.2
        ax.bar(xw, wr, width=0.4, label="WRB res", color="#4472C4")
        ax.bar(xh, hr, width=0.4, label="HFComp res", color="#ED7D31")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["LH", "HL", "LL", "HH"], fontsize=8)
        ax.legend(fontsize=8)
        ax.set_title("residual band composition (norm)", fontsize=10)
        fig.suptitle("Physical analysis [{}] (N={})".format(tag, len(rows)), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(out_dir, "physics_{}.png".format(tag)), bbox_inches="tight")
        plt.close(fig)
        print("  [fig] physics_{}.png".format(tag))


# ---- entry point ----
def main():
    ap = argparse.ArgumentParser(description="Physics proof of frequency-band modules for 15 models (3 tasks x 5 seeds)")
    ap.add_argument("--phase", type=str, default="both", choices=["train", "analyze", "both"])
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--tasks", type=str, default=None, help="comma-separated: A-B,A-C,B-C")
    ap.add_argument("--n_per_class", type=int, default=8, help="test samples per class")
    ap.add_argument("--gradcam", type=int, default=1, choices=[0, 1])
    ap.add_argument("--no_fig", type=int, default=0, choices=[0, 1])
    ap.add_argument("--force_retrain", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="print training commands only")
    args = ap.parse_args()
    global DRY_RUN
    DRY_RUN = bool(args.dry_run)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = [t for t in BEST_TASKS] if not args.tasks else \
        [t for t in BEST_TASKS if t["tag"] in [x.strip() for x in args.tasks.split(",")]]
    os.makedirs(OUT, exist_ok=True)

    if args.phase in ("train", "both"):
        phase_train(args, seeds, tasks)
    if args.phase in ("analyze", "both"):
        phase_analyze(args, seeds, tasks)


if __name__ == "__main__":
    main()

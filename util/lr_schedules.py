"""Per-epoch LR scaling for AdamW / generic optimizers, relative to initial param_group['lr'].
Linear / late_linear / cosine / warmup+cosine / onecycle schedules.
"""
from __future__ import annotations
import math


def linear_decay_scale(epoch_idx: int, total_epochs: int, eta_min_ratio: float = 0.05) -> float:
    """Full linear decay: scale goes from 1 to eta_min_ratio at epoch_idx (0-based). Constant 1 when total<=1."""
    te = int(total_epochs)
    if te <= 1:
        return 1.0
    em = float(max(0.0, min(1.0, eta_min_ratio)))
    t = float(epoch_idx) / float(max(1, te - 1))
    return 1.0 + (em - 1.0) * t


def late_linear_scale(
    epoch_idx: int, total_epochs: int, hold_frac: float, eta_min_ratio: float = 0.05
) -> float:
    """Hold full lr early, then linearly decay to eta_min_ratio. hold_frac in [0,1]: fraction of epochs at scale 1."""
    te = int(total_epochs)
    if te <= 1:
        return 1.0
    last_i = te - 1
    hf = float(max(0.0, min(1.0, hold_frac)))
    n_hold = min(last_i, int(round(last_i * hf)))
    em = float(max(0.0, min(1.0, eta_min_ratio)))
    if epoch_idx <= n_hold:
        return 1.0
    denom = max(1, last_i - n_hold)
    u = float(epoch_idx - n_hold) / float(denom)
    return 1.0 + (em - 1.0) * u


def cosine_scale(epoch_idx: int, total_epochs: int, eta_min_ratio: float = 0.05) -> float:
    """Full cosine annealing: scale from 1 to eta_min_ratio at epoch_idx (0-based). t=0 -> 1; t=1 -> eta_min_ratio."""
    te = int(total_epochs)
    if te <= 1:
        return 1.0
    em = float(max(0.0, min(1.0, eta_min_ratio)))
    t = float(epoch_idx) / float(te - 1)
    return em + (1.0 - em) * 0.5 * (1.0 + math.cos(math.pi * t))


def warmup_cosine_scale(
    epoch_idx: int, total_epochs: int, warmup_epochs: int = 3, eta_min_ratio: float = 0.05
) -> float:
    """Linear warmup from 0 to 1 over warmup_epochs (protects source features), then cosine anneal."""
    te = int(total_epochs)
    if te <= 1:
        return 1.0
    em = float(max(0.0, min(1.0, eta_min_ratio)))
    wu = max(1, int(warmup_epochs))
    if epoch_idx < wu:
        return float(epoch_idx + 1) / float(wu)
    rem = te - wu
    if rem <= 1:
        return em
    t = float(epoch_idx - wu) / float(rem - 1)
    return em + (1.0 - em) * 0.5 * (1.0 + math.cos(math.pi * t))


def one_cycle_scale(
    epoch_idx: int, total_epochs: int, peak_ratio: float = 1.0, eta_min_ratio: float = 0.05
) -> float:
    """1cycle: first 45% linear up from peak*0.1 to peak, last 55% cosine down to eta_min_ratio."""
    te = int(total_epochs)
    if te <= 1:
        return 1.0
    em = float(max(0.0, min(1.0, eta_min_ratio)))
    pk = float(max(0.05, peak_ratio))
    up = max(1, int(round(te * 0.45)))
    if epoch_idx < up:
        u = float(epoch_idx) / float(up)
        return pk * 0.1 + (pk - pk * 0.1) * u
    rem = te - up
    if rem <= 1:
        return em
    u = min(1.0, float(epoch_idx - up) / float(rem - 1))
    return em + (pk - em) * 0.5 * (1.0 + math.cos(math.pi * u))


def apply_lr_scale(optimizer, base_lrs: list, scale: float) -> None:
    """Set each param_group's lr to base_lrs[i] * scale."""
    s = float(scale)
    for i, pg in enumerate(optimizer.param_groups):
        pg["lr"] = float(base_lrs[i]) * s

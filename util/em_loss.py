"""Entropy-minimization loss family, extracted from loss_func.py."""
import torch
import torch.nn.functional as F


def entropy_loss(p_logit):
    """Entropy minimization; returns 0 on an empty batch to avoid div-by-zero."""
    if p_logit.size(0) == 0:
        return p_logit.new_tensor(0.0)
    p = F.softmax(p_logit, dim=-1)
    return -1 * torch.sum(p * F.log_softmax(p_logit, dim=-1)) / p_logit.size(0)


def entropy_loss_masked(p_logit, mask, eps=1e-8):
    """Entropy averaged only over samples where mask is True."""
    ent = -torch.sum(F.softmax(p_logit, dim=-1) * F.log_softmax(p_logit, dim=-1), dim=1)
    if not mask.any():
        return p_logit.new_tensor(0.0)
    return ent[mask].mean()


def entropy_loss_weighted(p_logit, weight, eps=1e-8):
    """Per-sample entropy, weighted mean by weight (used by SCW-EM self-consistent weighting)."""
    ent = -torch.sum(F.softmax(p_logit, dim=-1) * F.log_softmax(p_logit, dim=-1), dim=1)
    w = weight.clamp(min=eps)
    return (ent * w).sum() / w.sum()

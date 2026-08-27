"""CaCo category contrastive loss, extracted from loss_func.py. Supports CaN (CVPR 2024) asymmetric pair weights."""
import torch
import torch.nn.functional as F


def caco_catnce_loss(q, keys, key_labels, q_labels, num_classes, tau=0.07, eps=1e-8,
                     per_query_weight=None, asym_weight=None):
    """CaCo category contrastive loss with optional CaN (CVPR 2024) asymmetric pair weights.

    asym_weight: (C, C) matrix; asym_weight[i,j] increases the repulsion of query class i from key class j.
    """
    Bq, D = q.shape
    N = keys.size(0)
    if N == 0 or Bq == 0:
        return q.new_tensor(0.0)
    logits = (q @ keys.T) / tau  # [Bq, N]
    q_labels_exp = q_labels.unsqueeze(1)  # [Bq, 1]
    key_labels_exp = key_labels.unsqueeze(0)  # [1, N]
    pos_mask = (key_labels_exp == q_labels_exp).float()  # [Bq, N]
    valid = (pos_mask.sum(dim=1) > 0).float()
    if valid.sum() == 0:
        return q.new_tensor(0.0)
    logits_max = logits.max(dim=1, keepdim=True)[0].detach()
    logits_stable = logits - logits_max
    exp_logits = torch.exp(logits_stable)
    # CaN (CVPR 2024): asymmetric pair weights
    if asym_weight is not None:
        _aw = asym_weight.to(q.device)
        # pair_weights[i,j] = asym_weight[q_label[i], key_label[j]]
        _q_idx = q_labels.clamp(0, _aw.size(0) - 1).long()
        _k_idx = key_labels.clamp(0, _aw.size(1) - 1).long()
        _pw = _aw[_q_idx.unsqueeze(1), _k_idx.unsqueeze(0)]  # [Bq, N]
        # Weight negatives only; positives stay 1.0
        _w = torch.ones_like(exp_logits) + (_pw - 1.0) * (1.0 - pos_mask)
        exp_logits = exp_logits * _w.clamp_min(0.1)
    pos_sum = (exp_logits * pos_mask).sum(dim=1) + eps
    all_sum = exp_logits.sum(dim=1) + eps
    log_prob = torch.log(pos_sum) - torch.log(all_sum)
    neg_loglik = -log_prob * valid  # [Bq]
    if per_query_weight is not None:
        w = per_query_weight.view(-1).to(device=q.device, dtype=q.dtype).clamp(min=eps) * valid
        denom = w.sum().clamp_min(eps)
        return (neg_loglik * w).sum() / denom
    loss = neg_loglik.sum() / (valid.sum() + eps)
    return loss

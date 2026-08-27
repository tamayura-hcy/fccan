"""ANG: Batch ETF angular balance.

The original GCC module (weight-norm calibration / feature-norm balance FNB / batch ETF
angular balance ANG) was trimmed per the 2026-08 cleanup decision: only ANG remains
(lambda_batch_ang depends on it).

Usage:
    ang = AngModule(num_classes=3, feat_dim=1024)
    loss_ang = ang.angular_balance_loss(batch_means, pair_weights)  # batch ETF
"""

import torch
import torch.nn as nn


class AngModule(nn.Module):
    """Module keeping only the Batch ETF angular balance mechanism."""

    def __init__(self, num_classes: int, feat_dim: int = 1024, **__):
        super().__init__()
        self.num_classes = num_classes

    # ---- ANG: Batch ETF angular balance ----

    @staticmethod
    def angular_balance_loss(
        batch_means: torch.Tensor, pair_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Batch ETF equiangular constraint: off-diagonal cos -> target, variance -> 0.

        Parameters
        ----------
        batch_means : Tensor [K, D], L2-normalized per-class mean features
        pair_weights : Tensor [K, K] or None
            Adaptive weight matrix; off-diagonal = max(proto_cos[i,j] - target, 0).
            None means uniform weights (original behavior): closer prototypes get larger weights.

        Returns
        -------
        loss : scalar
            weighted mean((cos_ij - target)^2) + var(cos_ij)
        """
        K = batch_means.shape[0]
        if K < 2:
            return batch_means.new_tensor(0.0)
        cos_mat = batch_means @ batch_means.T
        target = -1.0 / max(K - 1, 1)
        mask = 1.0 - torch.eye(K, device=batch_means.device, dtype=batch_means.dtype)
        off_diag = cos_mat[mask.bool()]
        if pair_weights is not None:
            w_off = pair_weights[mask.bool()]
            return (w_off * (off_diag - target) ** 2).sum() / w_off.sum().clamp_min(1e-8) + off_diag.var()
        return ((off_diag - target) ** 2).mean() + off_diag.var()

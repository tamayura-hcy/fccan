# Herath et al., ICCV 2023 - SCAL (free-energy alignment) + SCON (score normalization).
# ResNet path: logits only, no ViT attention filtering.

from __future__ import annotations

import torch
import torch.nn as nn


def logits_to_energy(logits: torch.Tensor) -> torch.Tensor:
    """E(x, c) = -logit_c, shape [B, C] (per-class energy)."""
    return -logits


def free_energy_from_logits(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """F(x) = -logsumexp(tau * logits, dim=1), shape [B] (scalar free energy per sample)."""
    t = float(tau)
    return -torch.logsumexp(t * logits, dim=1)


class EnergyUdaState(nn.Module):
    """Running stats: EMA of F on source batches (for L_ea) and of F / per-class E per domain (for SCON).
    Momentum defaults to 0.1, matching the paper.
    """

    def __init__(self, num_classes: int, ema_momentum: float = 0.1, eps: float = 1e-5):
        """SCAL + SCON energy-alignment state."""
        super().__init__()
        C = int(num_classes)
        m = float(ema_momentum)
        self.eps = float(eps)
        self.C = C
        self.m = m
        # Source F batch-mean EMA -> used in max(0, F_t - mu_src)
        self.register_buffer("src_fe_ema", torch.zeros(()))
        # Domains 's' / 't': EMA mean/var of scalar F; EMA mean/var of vector E
        self.register_buffer("mu_fe_s", torch.zeros(()))
        self.register_buffer("var_fe_s", torch.ones(()))
        self.register_buffer("mu_fe_t", torch.zeros(()))
        self.register_buffer("var_fe_t", torch.ones(()))
        self.register_buffer("mu_e_s", torch.zeros(C))
        self.register_buffer("var_e_s", torch.ones(C))
        self.register_buffer("mu_e_t", torch.zeros(C))
        self.register_buffer("var_e_t", torch.ones(C))
        self.register_buffer("_inited", torch.zeros((), dtype=torch.bool))

    def _ema_update_scalar(self, cur_mu, cur_var, hat_mu, hat_var):
        if not bool(self._inited.item()):
            return hat_mu.detach(), hat_var.detach().clamp_min(self.eps)
        m = self.m
        return (
            m * cur_mu + (1.0 - m) * hat_mu.detach(),
            (m * cur_var + (1.0 - m) * hat_var.detach()).clamp_min(self.eps),
        )

    def _ema_update_vec(self, cur_mu, cur_var, hat_mu, hat_var):
        if not bool(self._inited.item()):
            return hat_mu.detach(), hat_var.detach().clamp_min(self.eps)
        m = self.m
        return (
            m * cur_mu + (1.0 - m) * hat_mu.detach(),
            (m * cur_var + (1.0 - m) * hat_var.detach()).clamp_min(self.eps),
        )

    def forward_losses(
        self,
        logits_s: torch.Tensor,
        logits_t: torch.Tensor,
        tau: float,
        lambda_scon: float,
        ea_weights: torch.Tensor = None,
    ):
        """logits_* : [B, C] unsoftmaxed.
        ea_weights : [Bt] per-sample L_ea weights (should be detached). None or all-1 -> plain mean.
        Returns (loss_ea, loss_scon, diag dict); everything without grad is detached.
        """
        C = logits_s.shape[1]
        if C != self.C:
            raise ValueError("num_classes mismatch EnergyUdaState")

        Fs = free_energy_from_logits(logits_s, tau)
        Ft = free_energy_from_logits(logits_t, tau)
        Es = logits_to_energy(logits_s)
        Et = logits_to_energy(logits_t)

        hat_mu_fs = Fs.mean()
        hat_var_fs = Fs.var(unbiased=False)
        hat_mu_ft = Ft.mean()
        hat_var_ft = Ft.var(unbiased=False)
        hat_mu_es = Es.mean(dim=0)
        hat_var_es = Es.var(dim=0, unbiased=False)
        hat_mu_et = Et.mean(dim=0)
        hat_var_et = Et.var(dim=0, unbiased=False)

        # Use current-batch stats for the normalization targets (mu,sigma from the previous EMA; first batch initializes)
        mu_s = self.mu_fe_s
        sig_s = (self.var_fe_s.clamp_min(self.eps)).sqrt()
        mu_t = self.mu_fe_t
        sig_t = (self.var_fe_t.clamp_min(self.eps)).sqrt()
        if not bool(self._inited.item()):
            mu_s, sig_s = hat_mu_fs.detach(), hat_var_fs.clamp_min(self.eps).detach().sqrt()
            mu_t, sig_t = hat_mu_ft.detach(), hat_var_ft.clamp_min(self.eps).detach().sqrt()

        z_fs = (Fs - mu_s.detach()) / (sig_s.detach().clamp_min(self.eps))
        z_ft = (Ft - mu_t.detach()) / (sig_t.detach().clamp_min(self.eps))
        L_fen_s = (Fs - z_fs).abs().mean()
        L_fen_t = (Ft - z_ft).abs().mean()

        mu_es = self.mu_e_s
        sig_es = (self.var_e_s.clamp_min(self.eps)).sqrt()
        mu_et = self.mu_e_t
        sig_et = (self.var_e_t.clamp_min(self.eps)).sqrt()
        if not bool(self._inited.item()):
            mu_es = hat_mu_es.detach()
            sig_es = hat_var_es.clamp_min(self.eps).detach().sqrt()
            mu_et = hat_mu_et.detach()
            sig_et = hat_var_et.clamp_min(self.eps).detach().sqrt()

        z_es = (Es - mu_es.detach()) / (sig_es.detach().clamp_min(self.eps).unsqueeze(0))
        z_et = (Et - mu_et.detach()) / (sig_et.detach().clamp_min(self.eps).unsqueeze(0))
        L_en_s = (Es - z_es).pow(2).mean()
        L_en_t = (Et - z_et).pow(2).mean()

        lam = float(lambda_scon)
        L_n_s = lam * L_fen_s + (1.0 - lam) * L_en_s
        L_n_t = lam * L_fen_t + (1.0 - lam) * L_en_t
        loss_scon = L_n_s + L_n_t

        # SCAL: max(0, F_t - tilde_mu_src); tilde_mu is the source-F batch-mean EMA
        mu_src_bar = self.src_fe_ema
        if not bool(self._inited.item()):
            mu_src_bar = hat_mu_fs.detach()
        ea_terms = torch.relu(Ft - mu_src_bar.detach())
        if ea_weights is None:
            loss_ea = ea_terms.mean()
        else:
            w = ea_weights.detach().to(ea_terms.dtype)
            loss_ea = (w * ea_terms).sum() / (w.sum() + 1e-8)

        diag = {
            "L_fen_s": L_fen_s.detach(),
            "L_fen_t": L_fen_t.detach(),
            "L_en_s": L_en_s.detach(),
            "L_en_t": L_en_t.detach(),
        }

        # Update EMA at the end of the training step (detached from backprop)
        if self.training:
            with torch.no_grad():
                new_src_ema = (
                    hat_mu_fs.detach()
                    if not bool(self._inited.item())
                    else self.m * self.src_fe_ema + (1.0 - self.m) * hat_mu_fs.detach()
                )
                self.src_fe_ema.copy_(new_src_ema)

                nmu_s, nvar_s = self._ema_update_scalar(self.mu_fe_s, self.var_fe_s, hat_mu_fs, hat_var_fs)
                nmu_t, nvar_t = self._ema_update_scalar(self.mu_fe_t, self.var_fe_t, hat_mu_ft, hat_var_ft)
                self.mu_fe_s.copy_(nmu_s)
                self.var_fe_s.copy_(nvar_s)
                self.mu_fe_t.copy_(nmu_t)
                self.var_fe_t.copy_(nvar_t)

                nmu_es, nvar_es = self._ema_update_vec(self.mu_e_s, self.var_e_s, hat_mu_es, hat_var_es)
                nmu_et, nvar_et = self._ema_update_vec(self.mu_e_t, self.var_e_t, hat_mu_et, hat_var_et)
                self.mu_e_s.copy_(nmu_es)
                self.var_e_s.copy_(nvar_es)
                self.mu_e_t.copy_(nmu_et)
                self.var_e_t.copy_(nvar_et)

                self._inited.fill_(True)

        return loss_ea, loss_scon, diag

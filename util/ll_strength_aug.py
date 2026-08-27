"""Low-frequency (LL) strength perturbation: break the "low-frequency strength = class" shortcut.

Rationale (from DWT-domain diagnostics): the domain shift / class shortcut lives mainly
in the LL subband (BOE AMD LL mean=0.459 vs TMI AMD=0.313; within BOE, AMD >> NORMAL),
so perturbing LL energy during source training forces the model to learn transferable
local morphology (drusen) instead.

Safety: only LL energy changes; LH/HL/HH structure and Haar sign are untouched, so
structure is preserved. Per-sample gain k=(1+alpha)^u is always positive.

Self-contained: ships its own correct Haar DWT/iDWT (independent of util/wavelet_recal.py)
to avoid changing the behavior of in-model modules (HFComp/WRB/WTConv); only affects
source training with --src_ll_aug enabled.
"""
import torch
import torch.nn.functional as F


def haar_dwt2d(x):
    """Correct 2D Haar DWT (per-channel subband layout, perfect reconstruction); self-contained."""
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, 'Haar DWT requires even H, W'
    w_ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32, device=x.device) / 2.0
    w_lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32, device=x.device) / 2.0
    w_hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32, device=x.device) / 2.0
    w_hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32, device=x.device) / 2.0
    base = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0)          # (4, 2, 2)
    weight = base.unsqueeze(0).expand(C, 4, 2, 2).reshape(4 * C, 2, 2).unsqueeze(1)
    out = F.conv2d(x, weight, stride=2, groups=C)                # (B, 4*C, H/2, W/2)
    # Output channel order [ch0:s0, ch0:s1, ...] -> split (C,4) then transpose to (4,C)
    out = out.view(B, C, 4, H // 2, W // 2).permute(0, 2, 1, 3, 4)
    return out


def haar_idwt2d_fixed(subbands):
    """Fixed 2D Haar iDWT.

    util.wavelet_recal.haar_idwt2d swaps the LH and HL synthesis kernels, which flips
    horizontal/vertical HF at reconstruction ([1,2,3,4] -> [1,3,2,4]). This version uses
    the correct orthogonal-Haar kernels (synthesis = analysis transpose, /2 normalized):
      LL: [[1,1],[1,1]]/2  LH: [[1,-1],[1,-1]]/2  HL: [[1,1],[-1,-1]]/2  HH: [[1,-1],[-1,1]]/2
    """
    B, _, C, h, w = subbands.shape
    device = subbands.device
    LL, LH, HL, HH = subbands[:, 0], subbands[:, 1], subbands[:, 2], subbands[:, 3]

    def _wt(k):
        k = torch.tensor(k, dtype=torch.float32, device=device).view(1, 1, 2, 2)
        return k.expand(C, 1, 2, 2).contiguous()

    w_ll = _wt([[1, 1], [1, 1]]) / 2.0
    w_lh = _wt([[1, -1], [1, -1]]) / 2.0   # fixed: correct LH synthesis kernel
    w_hl = _wt([[1, 1], [-1, -1]]) / 2.0   # fixed: correct HL synthesis kernel
    w_hh = _wt([[1, -1], [-1, 1]]) / 2.0

    y_ll = F.conv_transpose2d(LL, w_ll, stride=2, groups=C)
    y_lh = F.conv_transpose2d(LH, w_lh, stride=2, groups=C)
    y_hl = F.conv_transpose2d(HL, w_hl, stride=2, groups=C)
    y_hh = F.conv_transpose2d(HH, w_hh, stride=2, groups=C)
    return y_ll + y_lh + y_hl + y_hh


def ll_strength_augment(x, alpha=0.5, p=0.5, keep_dc=True, clamp_out=True):
    """LL-subband energy perturbation for a batch of images.

    Parameters
    ----------
    x : Tensor (B, C, H, W) image batch (usually [0,1]); H, W must be even.
    alpha : float, perturbation strength. k=(1+alpha)^u with u~U(-1,1), so
        k in [1/(1+alpha), 1+alpha], always positive; alpha=1 -> k in [0.5,2].
    p : float, application probability; otherwise returned unchanged.
    keep_dc : bool
        True  -> only contrast is perturbed (LL AC scaled), keeping brightness (DC) [default]
        False -> brightness and contrast both perturbed (breaks the LL shortcut harder)
    clamp_out : bool, clip the reconstruction back to the original range after iDWT.
    """
    if x.dim() != 4:
        raise ValueError('ll_strength_augment expects (B,C,H,W), got {}'.format(tuple(x.shape)))
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        return x
    if torch.rand(1).item() > p:
        return x

    sub = haar_dwt2d(x)              # (B, 4, C, H/2, W/2)  [LL, LH, HL, HH]
    LL = sub[:, 0]                   # (B, C, H/2, W/2)

    # Per-sample random LL gain (exponential: always positive, no degenerate/inverted images)
    # k=(1+alpha)^u, u~U(-1,1) -> k in [1/(1+alpha), 1+alpha], multiplicative-symmetric
    u = 2.0 * torch.rand(B, 1, 1, 1, device=x.device) - 1.0   # U(-1,1)
    k = (1.0 + alpha) ** u                                    # [B,1,1,1]

    if keep_dc:
        # Only scale AC (contrast), keep DC (brightness): LL_new = mean + k*(LL-mean)
        mu = LL.mean(dim=(2, 3), keepdim=True)
        LL_new = mu + k * (LL - mu)
    else:
        # Perturb brightness + contrast together (breaks the LL shortcut most thoroughly)
        LL_new = LL * k

    sub_aug = torch.stack([LL_new, sub[:, 1], sub[:, 2], sub[:, 3]], dim=1)
    x_aug = haar_idwt2d_fixed(sub_aug)  # (B, C, H, W) using the fixed iDWT

    if clamp_out and x_aug.min() >= 0.0 and x_aug.max() <= 1.0:
        x_aug = x_aug.clamp(0.0, 1.0)
    return x_aug


def ll_energy(x):
    """Per-image LL-subband energy (mean square) of shape (B,), for diagnostics."""
    sub = haar_dwt2d(x)
    LL = sub[:, 0]
    return (LL ** 2).mean(dim=(1, 2, 3))

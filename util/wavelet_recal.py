"""Wavelet modules: input-side helpers and frequency losses (backbone unchanged).

1. get_hf_guide(x): high-frequency guide map (B,3,H,W) from DWT for auxiliary supervision
2. wavelet_hf_consistency_loss: KL consistency between original and HF-guided predictions
3. WaveletRecalibrationBlock: [deprecated] original WRB, not recommended in the FEA backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_dwt_weights(device):
    """Fixed Haar 2D DWT filters, four 2x2 kernels: LL, LH, HL, HH."""
    w_ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32, device=device) / 2.0
    w_lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32, device=device) / 2.0
    w_hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32, device=device) / 2.0
    w_hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32, device=device) / 2.0
    return w_ll, w_lh, w_hl, w_hh


def haar_dwt2d(x):
    """2D Haar DWT: x (B, C, H, W) -> (B, 4, C, H/2, W/2); H, W must be even."""
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "Haar DWT requires even H, W"
    device = x.device
    w_ll, w_lh, w_hl, w_hh = _haar_dwt_weights(device)
    # Per-channel 4 2x2 convs with stride=2; groups=C, weight (4*C, 1, 2, 2) = 4 kernels [LL,LH,HL,HH]
    base = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0)  # (4, 2, 2)
    weight = base.unsqueeze(0).expand(C, 4, 2, 2).reshape(4 * C, 2, 2).unsqueeze(1)  # (4*C, 1, 2, 2)
    out = F.conv2d(x, weight, stride=2, groups=C)  # (B, 4*C, H/2, W/2)
    out = out.view(B, 4, C, H // 2, W // 2)
    return out


def haar_idwt2d(subbands):
    """2D Haar iDWT: (B, 4, C, H/2, W/2) -> (B, C, H, W)."""
    B, _, C, h, w = subbands.shape
    device = subbands.device
    LL, LH, HL, HH = subbands[:, 0], subbands[:, 1], subbands[:, 2], subbands[:, 3]
    # Inverse: per 2x2 block, rebuild 4 pixels from the 4 subband values via transposed conv (stride 2).
    w_ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32, device=device) / 2.0
    w_lh = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32, device=device) / 2.0
    w_hl = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32, device=device) / 2.0
    w_hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32, device=device) / 2.0
    weight_ll = w_ll.view(1, 1, 2, 2).expand(C, 1, 2, 2).contiguous()
    weight_lh = w_lh.view(1, 1, 2, 2).expand(C, 1, 2, 2).contiguous()
    weight_hl = w_hl.view(1, 1, 2, 2).expand(C, 1, 2, 2).contiguous()
    weight_hh = w_hh.view(1, 1, 2, 2).expand(C, 1, 2, 2).contiguous()
    # conv_transpose2d per subband: (B,C,h,w) -> (B,C,2h,2w)
    y_ll = F.conv_transpose2d(LL, weight_ll, stride=2, groups=C)
    y_lh = F.conv_transpose2d(LH, weight_lh, stride=2, groups=C)
    y_hl = F.conv_transpose2d(HL, weight_hl, stride=2, groups=C)
    y_hh = F.conv_transpose2d(HH, weight_hh, stride=2, groups=C)
    return y_ll + y_lh + y_hl + y_hh


def get_hf_guide(x, normalize=True):
    """High-frequency guide map from DWT for auxiliary supervision; backbone unchanged.

    x: (B, C, H, W) normalized to [0,1]; returns (B, C, H, W) with LH+HL+HH only, min-max to [0,1].
    """
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        return x.clone()
    subbands = haar_dwt2d(x)  # (B, 4, C, H/2, W/2)
    LL, LH, HL, HH = subbands[:, 0], subbands[:, 1], subbands[:, 2], subbands[:, 3]
    # HF only: zero out LL, then iDWT
    subbands_hf = torch.stack([
        torch.zeros_like(LL),
        LH, HL, HH
    ], dim=1)
    x_hf = haar_idwt2d(subbands_hf)  # (B, C, H, W)
    if normalize:
        # per-sample min-max to [0,1] to avoid negatives/out-of-range
        x_min = x_hf.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        x_max = x_hf.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        x_hf = (x_hf - x_min) / (x_max - x_min + 1e-8)
        x_hf = torch.clamp(x_hf, 0.0, 1.0)
    return x_hf


class WaveletRecalibrationBlock(nn.Module):
    """WRB: x -> DWT -> zero HH -> independent gates on LL/LH/HL -> iDWT -> residual fusion.

    Based on the WaveSDG finding + feature-map verification:
    - HH is noise at the feature-map level, so it is zeroed;
    - LL/LH/HL are out of sync at the feature-map level, so each gets its own gate.
    """
    def __init__(self, channels, alpha=0.4, lambda_init=0.3):
        super().__init__()
        self.alpha = alpha
        self.lambda_init = lambda_init
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.conv1x1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            return x
        subbands = haar_dwt2d(x)  # (B, 4, C, H/2, W/2)
        subbands[:, 3] = 0.0  # zero HH (WaveSDG finding + our verification)
        g = self.gate(subbands.reshape(B, -1, subbands.size(3), subbands.size(4)))  # (B, 4)
        scale = 1.0 + self.alpha * torch.tanh(g)  # (B, 4)
        scale = scale.view(B, 4, 1, 1, 1)
        subbands = subbands * scale
        x_rec = haar_idwt2d(subbands)
        lam = torch.sigmoid(self.lambda_param) * 0.5 + 0.25
        out = x + lam * self.conv1x1(x_rec)
        return out


class WaveletConvBranch(nn.Module):
    """Plan C: lightweight multiscale wavelet branch. 1-level DWT -> per-subband 3x3 depthwise conv -> iDWT -> add.
    Controlled by --use_wr_dagcn together with WRB (no new switch).
    """

    def __init__(self, channels, kernel_size=3, scale=0.2):
        super().__init__()
        self.channels = channels
        self.scale = scale
        self.dw_conv = nn.Conv2d(4 * channels, 4 * channels, kernel_size, padding=kernel_size // 2, groups=4 * channels)

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            return x * 0.0
        subbands = haar_dwt2d(x)
        subbands[:, 3] = 0.0  # zero HH
        h, w = subbands.size(3), subbands.size(4)
        y = subbands.reshape(B, 4 * C, h, w)
        y = self.dw_conv(y)
        y = y.reshape(B, 4, C, h, w)
        y = haar_idwt2d(y)
        return self.scale * y


class WTConvBlock(nn.Module):
    """Plan A: WTConv block, 1~2-level Haar + per-subband 3x3 depthwise conv + IWT, chained after WRB.
    1-level: DWT -> conv -> IWT; 2-level: cascade the LL again, Z(i)=IWT(Y_LL(i)+Z(i+1), Y_H(i)).
    """

    def __init__(self, channels, levels=2, kernel_size=3, scale=0.2):
        super().__init__()
        self.channels = channels
        self.levels = levels
        self.scale = scale
        pad = kernel_size // 2
        self.conv_level1 = nn.Conv2d(4 * channels, 4 * channels, kernel_size, padding=pad, groups=4 * channels)
        if levels >= 2:
            self.conv_level2 = nn.Conv2d(4 * channels, 4 * channels, kernel_size, padding=pad, groups=4 * channels)
        else:
            self.conv_level2 = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.levels == 1:
            if H % 2 != 0 or W % 2 != 0:
                return x * 0.0
            sub1 = haar_dwt2d(x)
            sub1[:, 3] = 0.0  # zero HH
            y1 = self.conv_level1(sub1.reshape(B, 4 * C, H // 2, W // 2)).reshape(B, 4, C, H // 2, W // 2)
            out = haar_idwt2d(y1)
            return self.scale * out
        # 2-level cascade
        if H % 4 != 0 or W % 4 != 0:
            return x * 0.0
        sub1 = haar_dwt2d(x)
        sub1[:, 3] = 0.0  # zero HH
        h1, w1 = H // 2, W // 2
        y1 = self.conv_level1(sub1.reshape(B, 4 * C, h1, w1)).reshape(B, 4, C, h1, w1)
        ll1 = y1[:, 0]
        sub2 = haar_dwt2d(ll1)
        sub2[:, 3] = 0.0  # zero HH
        h2, w2 = h1 // 2, w1 // 2
        y2 = self.conv_level2(sub2.reshape(B, 4 * C, h2, w2)).reshape(B, 4, C, h2, w2)
        z2 = haar_idwt2d(y2)
        y_ll1_plus_z2 = y1[:, 0] + z2
        combined1 = torch.stack([y_ll1_plus_z2, y1[:, 1], y1[:, 2], y1[:, 3]], dim=1)
        out = haar_idwt2d(combined1)
        return self.scale * out


class HFCompBlock(nn.Module):
    """
    High-frequency compensation branch:
    x -> DWT -> sum(LH, HL, HH) -> Conv+BN+ReLU+SE -> distribute back to LH/HL/HH -> IWT -> scaled residual.
    LL path is left unchanged and serves as the main structural branch.
    """

    def __init__(self, channels, reduction=4, scale=0.2):
        super().__init__()
        self.channels = channels
        self.scale = scale
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            return x * 0.0
        sub = haar_dwt2d(x)  # (B,4,C,h,w)
        LL, LH, HL, HH = sub[:, 0], sub[:, 1], sub[:, 2], sub[:, 3]
        # aggregate high-frequency bands
        hf = LH + HL  # HH zeroed
        y = self.conv(hf)
        w = self.se(y)
        y = y * w
        y_each = y / 2.0
        sub_hf = torch.stack([torch.zeros_like(LL), y_each, y_each, torch.zeros_like(HH)], dim=1)
        x_hf = haar_idwt2d(sub_hf)
        return self.scale * x_hf


def freq_band_dropout(images, p=0.5):
    """
    FAMNet (AAAI 2025) inspired: randomly drop one wavelet subband per image in the batch.
    Forces the model not to rely on a single band (NORMAL/DME sharing HF at 512).

    Args:
        images: (B, 3, H, W) tensor, normalized to [0,1]
        p: probability each image is processed
    Returns:
        augmented images, same shape
    """
    B = images.size(0)
    if B == 0:
        return images
    H, W = images.shape[2], images.shape[3]
    if H % 2 != 0 or W % 2 != 0:
        return images
    mask = torch.rand(B, device=images.device) < p
    if not mask.any():
        return images
    sub = haar_dwt2d(images[mask])  # (N, 4, 3, H/2, W/2)
    bands = torch.randint(0, 4, (mask.sum().item(),), device=images.device)
    for i, b in enumerate(bands):
        sub[i, b.item(), :, :, :] = 0.0
    rec = haar_idwt2d(sub)
    images_aug = images.clone()
    images_aug[mask] = rec
    return images_aug


def freq_profile(images):
    """
     Compute band-energy features of the input. DWT -> per-subband energy -> (B, 4).
     Used as context for frequency-aware calibration.
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
    sub = haar_dwt2d(images)  # (B, 4, C, h, w)
    energy = (sub ** 2).mean(dim=(2, 3, 4))  # (B, 4) per-band energy
    return energy

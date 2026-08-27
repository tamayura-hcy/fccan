import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from util.wavelet_recal import WaveletRecalibrationBlock, WTConvBlock, HFCompBlock, haar_dwt2d


class WaveletSpatialAttentionLite(nn.Module):
    """
    Lightweight MSW-SA style spatial attention:
    - Channel pooling (max + avg)
    - 1-level Haar DWT on pooled map, only LL is used (denoised, larger RF)
    - Small conv stack on LL -> 1-channel attention map
    - Upsample and apply residual spatial attention: x + x * S(F)
    """

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        # pooled has 2 channels (max + avg)
        self.conv_ll1 = nn.Conv2d(2, 8, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn_ll1 = nn.BatchNorm2d(8)
        self.conv_ll2 = nn.Conv2d(8, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        max_pool, _ = x.max(dim=1, keepdim=True)
        avg_pool = x.mean(dim=1, keepdim=True)
        pooled = torch.cat([max_pool, avg_pool], dim=1)  # (B,2,H,W)

        # 1-level Haar DWT on pooled map; fall back to bilinear downsample if odd size
        if H % 2 == 0 and W % 2 == 0:
            sub = haar_dwt2d(pooled)  # (B,4,2,h,w)
            LL = sub[:, 0]  # (B,2,h,w)
        else:
            LL = F.interpolate(pooled, scale_factor=0.5, mode='bilinear', align_corners=False)

        att = self.conv_ll1(LL)
        att = self.bn_ll1(att)
        att = F.relu(att, inplace=True)
        att = self.conv_ll2(att)  # (B,1,h,w)
        att = F.interpolate(att, size=(H, W), mode='bilinear', align_corners=False)
        att = torch.sigmoid(att)
        return x + x * att


class FEAWithWRB(nn.Module):
    """FEA branch: ResNet50 + dual WRB + WTConv/HFComp etc.

    conv1 -> ... -> layer2 -> WRB(512) -> [MSW-SA2] -> layer3 -> [MSW-SA3] -> WRB(1024) -> WTConv -> layer4 -> avgpool
    """

    def __init__(self, wrb_alpha=0.4, wrb_lambda=0.3, use_hf_comp=False, hf_comp_scale=0.2,
                 use_msw_sa=False, msw_sa_positions="23", use_wrb_after_layer2: bool = True):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        children = list(base.children())
        self.conv1 = children[0]
        self.bn1 = children[1]
        self.relu = children[2]
        self.maxpool = children[3]
        self.layer1 = children[4]
        self.layer2 = children[5]
        self.layer3 = children[6]
        self.layer4 = children[7]
        self.avgpool = children[8]
        self.wrb_early = (
            WaveletRecalibrationBlock(channels=512, alpha=wrb_alpha, lambda_init=wrb_lambda)
            if use_wrb_after_layer2 else None
        )
        self.wrb = WaveletRecalibrationBlock(channels=1024, alpha=wrb_alpha, lambda_init=wrb_lambda)
        self.wt_conv_branch = WTConvBlock(1024, levels=2, kernel_size=3, scale=0.2)
        self.use_hf_comp = use_hf_comp
        self.hf_comp = HFCompBlock(1024, reduction=4, scale=hf_comp_scale) if use_hf_comp else None

        self.use_msw_sa = use_msw_sa
        msw_positions = str(msw_sa_positions or "")
        self.msw_sa2 = WaveletSpatialAttentionLite(in_channels=512) if use_msw_sa and '2' in msw_positions else None
        self.msw_sa3 = WaveletSpatialAttentionLite(in_channels=1024) if use_msw_sa and '3' in msw_positions else None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.wrb_early is not None:
            x = self.wrb_early(x)
        if self.msw_sa2 is not None:
            x = self.msw_sa2(x)
        x = self.layer3(x)
        if self.msw_sa3 is not None:
            x = self.msw_sa3(x)
        x = self.wrb(x)
        x = x + self.wt_conv_branch(x)
        if self.use_hf_comp and self.hf_comp is not None:
            x = x + self.hf_comp(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x  # (B, 2048, 1, 1)

    def intermediate_gap_features(self, x):
        """Four GAP vectors (channels 256/512/1024/2048), aligned with FEANetBase.get_cnn_intermediates."""
        feats = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feats.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        x = self.layer2(x)
        if self.wrb_early is not None:
            x = self.wrb_early(x)
        if self.msw_sa2 is not None:
            x = self.msw_sa2(x)
        feats.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        x = self.layer3(x)
        if self.msw_sa3 is not None:
            x = self.msw_sa3(x)
        x = self.wrb(x)
        x = x + self.wt_conv_branch(x)
        if self.use_hf_comp and self.hf_comp is not None:
            x = x + self.hf_comp(x)
        feats.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        x = self.layer4(x)
        x = self.avgpool(x)
        feats.append(x.flatten(1))
        return feats


class FEANetBase(nn.Module):
    """Baseline encoder: plain ResNet50 (no FEA/WRB); used for the strict reference baseline."""

    def __init__(self):
        super(FEANetBase, self).__init__()
        self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.combined_features = features

    def get_cnn_intermediates(self, x):
        """Returns per-layer CNN features after GAP: 256, 512, 1024, 2048 dims."""
        # self.cnn = Sequential(conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4)
        x = self.cnn[0](x)
        x = self.cnn[1](x)
        x = self.cnn[2](x)
        x = self.cnn[3](x)
        feats = []
        for i in range(4, 8):
            x = self.cnn[i](x)
            feats.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        return feats

    def forward(self, x):
        features = self.cnn.forward(x)
        f = features.view(features.size(0), -1)
        return f, f


class FEANet(nn.Module):
    """FEA-Net (Frequency-Enhanced Attention Network).

    FEA: ResNet50 with a WRB after layer2 and layer3, outputting 2048-D features.
    forward returns (features, features) to keep the baseline interface.
    """

    def __init__(self, wrb_alpha=0.4, wrb_lambda=0.3, use_hf_comp=False, hf_comp_scale=0.2,
                 use_msw_sa=False, msw_sa_positions="23", use_wrb_after_layer2: bool = True):
        super(FEANet, self).__init__()
        # FEA branch: WRB + WTConv + optional HFComp/MSW-SA
        self.cnn = FEAWithWRB(wrb_alpha=wrb_alpha, wrb_lambda=wrb_lambda,
                              use_hf_comp=use_hf_comp, hf_comp_scale=hf_comp_scale,
                              use_msw_sa=use_msw_sa, msw_sa_positions=msw_sa_positions,
                              use_wrb_after_layer2=use_wrb_after_layer2)
        features = 2048
        self.combined_features = features

    def forward(self, x):
        # FEA-Net features
        features = self.cnn(x)
        f = features.view(features.size(0), -1)
        return f, f

    def get_cnn_intermediates(self, x):
        """WR-FEA multiscale GAP, dims match FEANetBase."""
        return self.cnn.intermediate_gap_features(x)


class Classifier(nn.Module):
    def __init__(self, features, num_classes, prob):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(features, 1024),
            nn.ReLU(),
            nn.Dropout(p=prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=prob),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        pseudo_label = self.classifier(x)
        mid_out = self.classifier[:-1](x)
        return pseudo_label, mid_out


class DiscriminatorBaseline(nn.Module):
    """Baseline discriminator: matches the reference, sigmoid output."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        super(DiscriminatorBaseline, self).__init__()
        self.restored = False
        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims, output_dims),
        )

    def forward(self, input):
        out = self.layer(input)
        out = torch.sigmoid(out)
        return out

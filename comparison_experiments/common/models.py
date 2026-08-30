"""共享模型：ResNet50 / VGG-16 backbone + 3 层分类器。

（Shared models: ResNet50/VGG-16 backbone + 3-layer classifier）

- 大部分基线用 ResNet50（thuml 官方 ResNet 配置、FEA-Net 同族）
- ADDA / EM-DDA 按 DAGCN 论文原文用 **VGG-16**（论文明确 "EM-DDA, ADDA, and other
  DA methods are implemented based on VGG-16, which has ~138M parameters"）
  —— 严格按论文复现时 backbone 不统一，论文里注明"各方法按原论文配置复现"。
"""
import torch
import torch.nn as nn


class Backbone(nn.Module):
    """ResNet50 (IMAGENET1K_V2) feature extractor, output 2048-d."""
    def __init__(self):
        super().__init__()
        from torchvision import models
        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.net.fc = nn.Identity()
        self.out_dim = 2048

    def forward(self, x):
        return self.net(x)


class ResNet18Backbone(nn.Module):
    """ResNet-18 (IMAGENET1K_V1) feature extractor, output 512-d.

    用于 source-only 基线（EM-DDA/DAGCN 论文对比了 ResNet-18 基线）。
    """
    def __init__(self):
        super().__init__()
        from torchvision import models
        self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Identity()
        self.out_dim = 512

    def forward(self, x):
        return self.net(x)


class VGGBackbone(nn.Module):
    """VGG-16 (IMAGENET1K_V1) feature extractor, output 4096-d (fc7).

    与 DAGCN 论文一致：ADDA / EM-DDA 等 DA 方法基于 VGG-16（~138M 参数）。
    """
    def __init__(self):
        super().__init__()
        from torchvision import models
        net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        net.classifier[6] = nn.Identity()   # 去掉 1000 类输出，取 fc7 → 4096 维
        self.net = net
        self.out_dim = 4096

    def forward(self, x):
        return self.net(x)


class VGGBnBackbone(nn.Module):
    """OCT-DDA 官方 ADDA/EM-DDA 编码器（xuqing88/OCT_DDA 官方代码）。

    官方：`src_encoder = vgg16_bn().features`（卷积特征，展平 25088 = 512*7*7 维），
    判别器作用在这 25088 维原始卷积特征上（非 fc7）。
    """
    def __init__(self):
        super().__init__()
        from torchvision import models
        net = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.net = net.features
        self.out_dim = 512 * 7 * 7  # 25088

    def forward(self, x):
        return self.net(x)   # [B, 512, 7, 7]


class VGGBnClassifier(nn.Module):
    """OCT-DDA 官方分类头：vgg16_bn 原生 classifier（fc6-fc7-fc8'，fc8'=num_classes）。"""
    def __init__(self, num_classes=3):
        super().__init__()
        from torchvision import models
        net = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        features = list(net.classifier.children())[:-1]   # 去掉 fc8(1000)
        features.extend([nn.Linear(net.classifier[6].in_features, num_classes)])
        self.net = nn.Sequential(*features)

    def forward(self, x):
        if x.dim() > 2:   # 兼容 test() 不展平：自动把 [B,512,7,7] 展平为 [B,25088]
            x = x.view(x.size(0), -1)
        return self.net(x)


class VGGClassifier(nn.Module):
    """VGG-16 原生分类头（论文 EM-DDA/ADDA：3 个 FC 层，最后输出 3 类）。

    VGGBackbone 已含 fc6/fc7（输出 4096-d fc7 特征），这里补 fc7'→fc8'：
    Linear(4096, 4096) + ReLU + Dropout(0.5) + Linear(4096, num_classes)，
    结构对应 VGG-16 原生 classifier 的后两层。
    """
    def __init__(self, in_dim=4096, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        return self.net(x)


def _make_bottleneck(in_dim, bottleneck_dim, style='bn'):
    """官方 bottleneck 结构：
    - 'bn'（thuml DANN/CDAN 官方）：Linear + BN + ReLU
    - 'dropout'（thuml DAN 官方）：Linear + ReLU + Dropout(0.5)
    """
    if style == 'dropout':
        return nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
    return nn.Sequential(
        nn.Linear(in_dim, bottleneck_dim),
        nn.BatchNorm1d(bottleneck_dim),
        nn.ReLU(inplace=True))


class Bottleneck(nn.Module):
    """thuml 官方 bottleneck：Linear(in, 256) + [BN] + ReLU（modules/classifier.py ImageClassifier）。"""
    def __init__(self, in_dim=2048, bottleneck_dim=256, style='bn'):
        super().__init__()
        self.net = _make_bottleneck(in_dim, bottleneck_dim, style)

    def forward(self, x):
        return self.net(x)


class ThumlClassifier(nn.Module):
    """thuml 官方 ImageClassifier：bottleneck(256) + head(Linear(256, C))。

    - forward(x)：返回 logits（评估/通用调用兼容）
    - forward_features(x)：返回 bottleneck 后 256 维特征（DANN/CDAN/MMD 用）
    - predict_from_feature(f)：从 bottleneck 特征出 logits（CDAN 用）
    - get_parameters：分层 lr，backbone 0.1×，bottleneck/head 1×（官方）
    bottleneck_style：'bn'（DANN/CDAN）或 'dropout'（DAN 官方）。
    """
    def __init__(self, in_dim=2048, num_classes=3, bottleneck_dim=256, bottleneck_style='bn'):
        super().__init__()
        self.bottleneck = Bottleneck(in_dim, bottleneck_dim, bottleneck_style)
        self.head = nn.Linear(bottleneck_dim, num_classes)
        self.features_dim = bottleneck_dim

    def forward(self, x):
        return self.head(self.bottleneck(x))

    def forward_features(self, x):
        return self.bottleneck(x)

    def predict_from_feature(self, f):
        return self.head(f)

    def get_parameters(self, base_lr=1.0, backbone=None):
        params = []
        if backbone is not None:
            params.append({"params": backbone.parameters(), "lr": 0.1 * base_lr})
        params.append({"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr})
        params.append({"params": self.head.parameters(), "lr": 1.0 * base_lr})
        return params


class Classifier(nn.Module):
    """旧版 3 层大 MLP（仅供 VGG 等特殊用途保留；ResNet 系一律用 ThumlClassifier）。"""
    def __init__(self, in_dim=2048, num_classes=3, prob=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(prob),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(prob),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        return self.net(x)


def build_models(num_classes, device, backbone='resnet50', bottleneck_dim=256,
                 bottleneck_style='bn'):
    """返回 (encoder, classifier)，均已 .to(device)。

    backbone: 'resnet50'（默认）/ 'resnet18' / 'vgg16'（论文 ADDA/EM-DDA 配置）
    bottleneck_style: 'bn'（DANN/CDAN 官方）/'dropout'（DAN 官方）
    """
    if backbone == 'vgg16':
        enc = VGGBackbone().to(device)
        clf = VGGClassifier(in_dim=4096, num_classes=num_classes).to(device)
    elif backbone == 'resnet18':
        enc = ResNet18Backbone().to(device)
        clf = ThumlClassifier(in_dim=512, num_classes=num_classes,
                              bottleneck_dim=bottleneck_dim,
                              bottleneck_style=bottleneck_style).to(device)
    else:
        enc = Backbone().to(device)
        clf = ThumlClassifier(in_dim=2048, num_classes=num_classes,
                              bottleneck_dim=bottleneck_dim,
                              bottleneck_style=bottleneck_style).to(device)
    return enc, clf

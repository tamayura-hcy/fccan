"""Shared models: ResNet50 / VGG-16 backbone + 3-layer classifier.

Most baselines use ResNet50 (thuml official ResNet config).
ADDA / EM-DDA use VGG-16 per the DAGCN paper ("EM-DDA, ADDA, and other DA methods
are implemented based on VGG-16, which has ~138M parameters").
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

    Used for the source-only baseline (EM-DDA/DAGCN papers compare a ResNet-18 baseline).
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

    Per the DAGCN paper, ADDA / EM-DDA are based on VGG-16 (~138M params).
    """
    def __init__(self):
        super().__init__()
        from torchvision import models
        net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        net.classifier[6] = nn.Identity()   # drop the 1000-class head, keep fc7 -> 4096-d
        self.net = net
        self.out_dim = 4096

    def forward(self, x):
        return self.net(x)


class VGGBnBackbone(nn.Module):
    """OCT-DDA official ADDA/EM-DDA encoder (xuqing88/OCT_DDA).

    Official: src_encoder = vgg16_bn().features (conv features flattened to 25088 = 512*7*7),
    discriminator operates on these raw conv features (not fc7).
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
    """OCT-DDA official classifier head: vgg16_bn native classifier (fc6-fc7-fc8', fc8'=num_classes)."""
    def __init__(self, num_classes=3):
        super().__init__()
        from torchvision import models
        net = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        features = list(net.classifier.children())[:-1]   # drop fc8 (1000 classes)
        features.extend([nn.Linear(net.classifier[6].in_features, num_classes)])
        self.net = nn.Sequential(*features)

    def forward(self, x):
        if x.dim() > 2:   # compat with test(): flatten [B,512,7,7] to [B,25088]
            x = x.view(x.size(0), -1)
        return self.net(x)


class VGGClassifier(nn.Module):
    """VGG-16 classifier head (paper EM-DDA/ADDA: 3 FC layers, output 3 classes).

    VGGBackbone already contains fc6/fc7 (4096-d fc7 features); this adds fc7'->fc8':
    Linear(4096, 4096) + ReLU + Dropout(0.5) + Linear(4096, num_classes).
    """
    def __init__(self, in_dim=4096, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        return self.net(x)


def _make_bottleneck(in_dim, bottleneck_dim, style='bn'):
    """Official bottleneck:
    - 'bn' (thuml DANN/CDAN): Linear + BN + ReLU
    - 'dropout' (thuml DAN): Linear + ReLU + Dropout(0.5)
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
    """thuml official bottleneck: Linear(in, 256) + [BN] + ReLU (modules/classifier.py ImageClassifier)."""
    def __init__(self, in_dim=2048, bottleneck_dim=256, style='bn'):
        super().__init__()
        self.net = _make_bottleneck(in_dim, bottleneck_dim, style)

    def forward(self, x):
        return self.net(x)


class ThumlClassifier(nn.Module):
    """thuml official ImageClassifier: bottleneck(256) + head(Linear(256, C)).

    forward(x): logits; forward_features(x): 256-d bottleneck features (DANN/CDAN/MMD);
    predict_from_feature(f): logits from bottleneck features (CDAN);
    get_parameters: layered lr, backbone 0.1x, bottleneck/head 1x (official).
    bottleneck_style: 'bn' (DANN/CDAN) or 'dropout' (DAN official).
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
    """Legacy 3-layer MLP (kept for VGG etc.; ResNet-based methods use ThumlClassifier)."""
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
    """Return (encoder, classifier), both on device.

    backbone: 'resnet50' (default) / 'resnet18' / 'vgg16' (paper ADDA/EM-DDA config)
    bottleneck_style: 'bn' (DANN/CDAN official) / 'dropout' (DAN official)
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

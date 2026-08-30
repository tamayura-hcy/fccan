# -*- coding: utf-8 -*-
"""stats_params.py —— 统计对比方法推理模型的参数量（M）。"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count(model):
    return sum(p.numel() for p in model.parameters())


def stats_adda():
    sys.path.insert(0, os.path.join(ROOT, "comparison_experiments"))
    from comparison_experiments.common.models import VGGBnBackbone, VGGBnClassifier
    enc = VGGBnBackbone()
    clf = VGGBnClassifier(num_classes=3)
    return count(enc) + count(clf)


def stats_dagcn():
    sys.path.insert(0, os.path.join(ROOT, "comparison_experiments", "dagcn"))
    from model import DAGCNModel, Classifier
    m = DAGCNModel()
    c = Classifier(m.combined_features, 3)
    return count(m) + count(c)


def stats_tvt():
    sys.path.insert(0, os.path.join(ROOT, "comparison_experiments",
                                    "third_party", "TVT"))
    from models.modeling import VisionTransformer, CONFIGS, AdversarialNetwork
    cfg = CONFIGS["ViT-B_16"]
    m = VisionTransformer(cfg, 224, zero_head=True, num_classes=3, msa_layer=12)
    ad = AdversarialNetwork(cfg.hidden_size // 12, cfg.hidden_size // 12)
    return count(m) + count(ad)


def stats_dac():
    sys.path.insert(0, os.path.join(ROOT, "comparison_experiments",
                                    "third_party", "DaC", "VisDA"))
    import network
    netF = network.ResBase(res_name="resnet50")
    netB = network.feat_bootleneck(type="bn", feature_dim=netF.in_features,
                                   bottleneck_dim=256)
    netC = network.feat_classifier(type="wn", class_num=3, bottleneck_dim=256)
    return count(netF) + count(netB) + count(netC)


if __name__ == "__main__":
    print("ADDA : {:.2f} M".format(stats_adda() / 1e6))
    print("DAGCN: {:.2f} M".format(stats_dagcn() / 1e6))
    print("TVT  : {:.2f} M".format(stats_tvt() / 1e6))
    print("DaC  : {:.2f} M".format(stats_dac() / 1e6))

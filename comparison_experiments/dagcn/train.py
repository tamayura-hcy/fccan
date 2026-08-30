"""DAGCN（Domain-Adaptive Graph Convolutional Network）baseline。

Tao et al., "Cross-Domain Retinopathy Classification Based on OCT Sensors
via DAGCN", IEEE TMI 2025. Code adapted from the official DAGCN project.

统一协议（与 comparison_experiments 其他方法一致）：同一 load_task 数据划分、
同一 input_size=256、同一种子。DAGCN 模型 = 双 ResNet50 + GCN(2112 维)。

★ 损失公式照论文 Eq.(8)：L = L_Adv + η·L_EM + λ·L_DA + γ·L_CA，η=1，λ=γ=0.03（统一）。
  【2026-08-09 回退】曾按论文任务特定 λ/γ（A-B:0.005/0.01, A-C:0.001/0.01, B-C:0.001/0.1）实测：
  A-C 略降（0.7896 vs 0.7964）、B-C 引入崩溃（s123 gmean=0）→ 无益，回退统一 0.03/0.03。
  对抗损失用 concat 版本（pred_domain_concat × pseudo_domain_concat）。
  classifier lr 分任务：A->B/A->C=0.001，B->C=0.0001（官方注释）。

流程：
  1) train_src：源域 CE + 0.001·structure-aware alignment（scores 上）
  2) tgt_encoder 复制源权重
  3) train_tgt：域判别器 + 目标编码器对抗
     （loss = loss_tgt + 1·loss_em + λ·loss_da + γ·loss_ca，论文 Eq.8，λ/γ 按任务）

依赖：torch_geometric（pip install torch_geometric）

用法：
    python -m comparison_experiments.dagcn.train --src A --tgt B --seed 777
"""
import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.common.data_loader import load_task, set_seed, LABEL_TO_DATASET
from comparison_experiments.common.evaluate import test
from comparison_experiments.dagcn.model import DAGCNModel, Classifier, Discriminator


# ───────────── 损失（移植自官方 util/loss_func.py）─────────────
def entropy_loss(p_logit):
    p = torch.softmax(p_logit, dim=-1)
    return -1 * torch.sum(p * torch.log_softmax(p_logit, dim=-1)) / p_logit.size()[0]


def uda_domain_alignment_loss(domain_pred, domain_target):
    _, domain_pred = domain_pred.max(1)
    loss_function = nn.BCELoss()
    return loss_function(domain_pred.to(torch.float), domain_target.to(torch.float))


def uda_structure_aware_alignment_loss(scores, classes, threshold=1):
    """结构感知对齐：同类样本距离 < 异类样本距离（triplet 式）。"""
    scores_copy = scores.clone()
    classes_copy = classes.clone().detach().cpu().numpy()
    unique, counts = np.unique(classes_copy, return_counts=True)
    class_count_dict = dict(zip(unique, counts))

    source_cat = -1
    counter = 0
    for key in class_count_dict:
        counter += 1
        if source_cat == -1:
            if class_count_dict[key] > 1:
                source_cat = key
        else:
            if class_count_dict[key] > 1 and random.random() > 0.01:
                source_cat = key
    if counter < 2 or source_cat == -1:
        return 0

    first_sample = second_sample = third_sample = None
    for i in range(len(scores_copy)):
        if classes_copy[i] == source_cat:
            if first_sample is None:
                first_sample = scores_copy[i]
            elif second_sample is None:
                second_sample = scores_copy[i]
            else:
                choice = random.choice([0, 1])
                n = random.random()
                if n > 0.5:
                    if choice == 0:
                        first_sample = scores_copy[i]
                    else:
                        second_sample = scores_copy[i]
        else:
            if third_sample is None:
                third_sample = scores_copy[i]
            else:
                n = random.random()
                if n > 0.5:
                    third_sample = scores_copy[i]
    if first_sample is None or second_sample is None or third_sample is None:
        return 0

    same_class_squared_dist = sum((first_sample - second_sample) ** 2)
    diff_class_squared_dist = sum((first_sample - third_sample) ** 2)
    l = same_class_squared_dist - diff_class_squared_dist + threshold
    return max(l, 0)


def uda_class_alignment_loss(x_src, x_tgt, pseudo_classes, classes):
    """类原型对齐：源/目标同类别原型距离（平方欧氏）之和。"""
    x_source_copy = x_src.clone()
    x_target_copy = x_tgt.clone()
    pseudo_classes_target_copy = torch.argmax(pseudo_classes.clone(), dim=1)
    classes_source_copy = classes.clone()

    source_dict = dict(zip(x_source_copy, classes_source_copy))
    final_source_dict = {}
    for key in source_dict:
        counter = 1
        total = key
        for inner_key in source_dict:
            if not torch.all(torch.eq(key, inner_key)) and source_dict[key].item() == source_dict[inner_key].item():
                counter += 1
                total = total + inner_key
        final_source_dict[source_dict[key].item()] = total / counter

    target_dict = dict(zip(x_target_copy, pseudo_classes_target_copy))
    final_target_dict = {}
    for key in target_dict:
        counter = 1
        total = key
        for inner_key in target_dict:
            if not torch.all(torch.eq(key, inner_key)) and target_dict[key].item() == target_dict[inner_key].item():
                counter += 1
                total = total + inner_key
        final_target_dict[target_dict[key].item()] = total / counter

    sum_dists = 0
    for key in final_source_dict:
        if key in final_target_dict:
            sum_dists = sum_dists + ((final_source_dict[key] - final_target_dict[key]) ** 2).sum(axis=0)
    return sum_dists


# ───────────── 训练流程（移植自官方 main.py）─────────────
def train_src(encoder, classifier, src_loader, device, epochs, lr=0.01):
    optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()),
                          lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epochs):
        encoder.train()
        classifier.train()
        for inputs, labels in src_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            features, scores = encoder(inputs)
            outputs = classifier(features)[0]
            loss_c = criterion(outputs, labels)
            loss_t = uda_structure_aware_alignment_loss(scores, labels)
            loss = loss_c + 0.001 * loss_t
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("  [DAGCN] source stage epoch {}/{} loss={:.4f}".format(ep + 1, epochs, loss.item()))


def train_tgt(src_encoder, classifier, tgt_encoder, netD, src_loader, tgt_loader,
              device, epochs, classifier_lr=0.001):
    criterion = nn.CrossEntropyLoss()
    # classifier lr：A->B/A->C=0.001，B->C=0.0001（由调用方按任务传入）
    optimizer_tgt = optim.SGD([{'params': classifier.parameters(), 'lr': classifier_lr},
                               {'params': tgt_encoder.parameters()}],
                              lr=0.001, momentum=0.9)
    optimizer_critic = optim.SGD(netD.parameters(), lr=0.01, momentum=0.9)

    src_encoder.eval()
    tgt_encoder.train()
    classifier.train()
    netD.train()

    for epoch in range(epochs):
        it = zip(src_loader, tgt_loader)
        for step, ((images_src, label_src), (images_tgt, label_tgt)) in enumerate(it):
            if torch.cuda.is_available():
                images_src, images_tgt = images_src.cuda(), images_tgt.cuda()
                label_src, label_tgt = label_src.cuda(), label_tgt.cuda()
            bs_src, bs_tgt = images_src.size(0), images_tgt.size(0)

            # ── 2.1 训练判别器 ──
            optimizer_critic.zero_grad()
            feat_src = src_encoder(images_src)[0].detach()
            feat_tgt = tgt_encoder(images_tgt)[0].detach()
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            pred_concat = netD(feat_concat)
            domain_src = torch.zeros(bs_src, dtype=torch.long, device=feat_src.device)
            domain_tgt = torch.ones(bs_tgt, dtype=torch.long, device=feat_tgt.device)
            domain_concat = torch.cat((domain_src, domain_tgt), 0)
            loss_critic = criterion(pred_concat, domain_concat)
            loss_critic.backward()
            optimizer_critic.step()

            # ── 2.2 训练目标编码器（对抗 + 熵 + 类对齐 + 域对齐）──
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()
            feat_src, scores_src = src_encoder(images_src)
            feat_tgt, scores_tgt = tgt_encoder(images_tgt)
            # ★ 关键：必须在此重新计算 feat_concat（非 detach），
            #   否则 pred_domain_concat/loss_tgt 复用的是 2.1 里 .detach() 的版本，
            #   对抗梯度会断掉 → 目标编码器只剩熵最小化驱动 → 类别塌缩（gmean=0）。
            #   官方 main.py train_tgt 2.2 节即如此（李生核对过官方代码）。
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            preds_tgt, mid_out_tgt = classifier(feat_tgt)
            preds_src, mid_out_src = classifier(feat_src)

            pred_domain_src = netD(feat_src)
            pred_domain_tgt = netD(feat_tgt)
            pred_domain_concat = netD(feat_concat)

            pseudo_domain_src = torch.ones(bs_src, dtype=torch.long, device=feat_src.device)
            pseudo_domain_tgt = torch.zeros(bs_tgt, dtype=torch.long, device=feat_tgt.device)
            pseudo_domain_concat = torch.cat((pseudo_domain_src, pseudo_domain_tgt), 0)

            loss_em = entropy_loss(preds_tgt)
            loss_da = uda_domain_alignment_loss(pred_domain_concat, domain_concat)
            loss_tgt = criterion(pred_domain_concat, pseudo_domain_concat)
            loss_ca = uda_class_alignment_loss(mid_out_src, mid_out_tgt, preds_tgt, label_src)
            # 论文 Eq.(8)：L = L_Adv + η·L_EM + λ·L_DA + γ·L_CA（η=1，λ=γ=0.03 统一）
            loss = loss_tgt + 1 * loss_em + 0.03 * loss_da + 0.03 * loss_ca
            loss.backward()
            optimizer_tgt.step()

        print("  [DAGCN] epoch {}/{} loss={:.4f} (em={:.3f} ca={:.3f} da={:.3f})".format(
            epoch + 1, epochs, loss.item(), loss_em.item(), loss_ca, loss_da.item()))


def train(args):
    set_seed(args.seed)
    # DAGCN 官方代码：仅 Resize+CenterCrop，无任何数据增强、无 Normalize（train_aug=False, normalize=False）
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch, train_aug=False,
                     normalize=False)
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))
    print("[DAGCN] 官方配置：无增强 / input {} / batch {} / src_epochs {}".format(
        args.input_size, args.batch, args.src_epochs))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[DAGCN] device={}  building DAGCN model (dual ResNet50 + GCN) ...".format(device), flush=True)
    src_encoder = DAGCNModel().to(device)
    classifier = Classifier(src_encoder.combined_features, n_cls).to(device)
    print("[DAGCN] models ready (feature dim={})".format(src_encoder.combined_features), flush=True)

    print("  [DAGCN] stage 1: source training")
    train_src(src_encoder, classifier, data['src_train'], device,
              epochs=args.src_epochs, lr=args.lr)

    tgt_encoder = DAGCNModel().to(device)
    tgt_encoder.load_state_dict(src_encoder.state_dict())
    netD = Discriminator(input_dims=src_encoder.combined_features, hidden_dims=500,
                         output_dims=2).to(device)

    # classifier lr：A->B/A->C=0.001，B->C=0.0001（学长/官方注释）
    classifier_lr = 0.0001 if (args.src == 'B' and args.tgt == 'C') else 0.001
    print("  [DAGCN] stage 2: target adversarial adaptation (classifier_lr={}, λ=γ=0.03)".format(
        classifier_lr))
    train_tgt(src_encoder, classifier, tgt_encoder, netD, data['src_train'],
              data['tgt_train'], device, epochs=args.epochs, classifier_lr=classifier_lr)

    acc, auc, _ = test(tgt_encoder, classifier, data['tgt_test'], len(data['tgt_test'].dataset),
                       num_classes=n_cls, class_names=data['class_names'])
    print("  [DAGCN] final tgt_acc={:.4f} tgt_auc={:.4f}".format(acc, auc))


def main():
    parser = argparse.ArgumentParser(description='DAGCN baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--src_epochs', type=int, default=15,
                        help='DAGCN 官方源域 15 epochs')
    parser.add_argument('--epochs', type=int, default=10,
                        help='目标域 epochs（官方 help：B->C 建议 15）')
    parser.add_argument('--batch', type=int, default=32,
                        help='DAGCN 官方每域 batch 32')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--input_size', type=int, default=224,
                        help='DAGCN 官方 Resize(256)->CenterCrop(224)')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

"""DAGCN (Domain-Adaptive Graph Convolutional Network) baseline.

Tao et al., "Cross-Domain Retinopathy Classification Based on OCT Sensors via DAGCN",
IEEE TMI 2025, adapted from the official DAGCN code.
Unified protocol (same as other comparison methods): same load_task splits, same seeds.
Model = dual ResNet50 + GCN (2112-d). Loss follows Eq.(8): L_Adv + n*L_EM + l*L_DA
+ g*L_CA (n=1, l=g=0.03). Flow: train_src, copy weights, then train_tgt.

Requires torch_geometric.
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


# Loss functions (ported from official util/loss_func.py)
def entropy_loss(p_logit):
    p = torch.softmax(p_logit, dim=-1)
    return -1 * torch.sum(p * torch.log_softmax(p_logit, dim=-1)) / p_logit.size()[0]


def uda_domain_alignment_loss(domain_pred, domain_target):
    _, domain_pred = domain_pred.max(1)
    loss_function = nn.BCELoss()
    return loss_function(domain_pred.to(torch.float), domain_target.to(torch.float))


def uda_structure_aware_alignment_loss(scores, classes, threshold=1):
    """Structure-aware alignment: same-class distance < cross-class distance (triplet-like)."""
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
    """Class-prototype alignment: sum of squared Euclidean distances between same-class source/target prototypes."""
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


# Training flow (ported from official main.py)
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
    # classifier lr: A->B/A->C=0.001, B->C=0.0001 (passed by caller per task)
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

            # -- 2.1 train discriminator --
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

            # -- 2.2 train target encoder (adversarial + entropy + class + domain alignment) --
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()
            feat_src, scores_src = src_encoder(images_src)
            feat_tgt, scores_tgt = tgt_encoder(images_tgt)
            # Important: recompute feat_concat without detach here, otherwise the 2.1
            # detached version is reused and the adversarial gradient breaks, leaving
            # only entropy minimization to drive the target encoder (class collapse).
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
            # Eq.(8): L = L_Adv + n*L_EM + l*L_DA + g*L_CA (n=1, l=g=0.03)
            loss = loss_tgt + 1 * loss_em + 0.03 * loss_da + 0.03 * loss_ca
            loss.backward()
            optimizer_tgt.step()

        print("  [DAGCN] epoch {}/{} loss={:.4f} (em={:.3f} ca={:.3f} da={:.3f})".format(
            epoch + 1, epochs, loss.item(), loss_em.item(), loss_ca, loss_da.item()))


def train(args):
    set_seed(args.seed)
    # official DAGCN: Resize+CenterCrop only, no aug, no Normalize
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch, train_aug=False,
                     normalize=False)
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))
    print("[DAGCN] official config: no augmentation / input {} / batch {} / src_epochs {}".format(
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

    # classifier lr: A->B/A->C=0.001, B->C=0.0001 (per official notes)
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
                        help='DAGCN official source 15 epochs')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Target epochs (official help: B->C suggest 15)')
    parser.add_argument('--batch', type=int, default=32,
                        help='DAGCN official per-domain batch 32')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--input_size', type=int, default=224,
                        help='DAGCN official Resize(256)->CenterCrop(224)')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

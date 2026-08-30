"""SHOT (Source Hypothesis Transfer) baseline.

Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer
for Unsupervised Domain Adaptation", ICML 2020.
按 tim-learn/SHOT 官方复现（object/image_source.py + image_target.py）：
  - 结构：ResNet50(2048) + feat_bottleneck(256,BN) + feat_classifier(linear)
  - 源训练：CrossEntropyLabelSmooth(ε=0.1)，分层 lr（netF 0.1×lr, netB/netC 1×lr）
  - 目标适应：聚类伪标签（obtain_label，cosine 距离 2 轮）+ CE(cls_par=0.3)
              + 熵最小化(ent_par=1.0) + 全局熵(gent)
  - lr_scheduler：lr*(1+10*iter/max)^(-0.75), wd=1e-3, nesterov=True；lr=1e-2

Usage:
    python -m comparison_experiments.shot.train --src A --tgt B --seed 777
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comparison_experiments.common.data_loader import load_task, set_seed, LABEL_TO_DATASET
from comparison_experiments.common.evaluate import test
from comparison_experiments.common.models import build_models


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    """SHOT 官方 lr_scheduler：lr0 * (1+gamma*iter/max)^(-power)，wd=1e-3，nesterov。"""
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class CrossEntropyLabelSmooth(nn.Module):
    """SHOT 官方源训练损失：label smoothing CE（loss.py CrossEntropyLabelSmooth）。"""

    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        # 官方：先建 CPU one-hot 再 scatter（避免 index/张量设备不一致），最后移到 inputs 设备
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if targets.is_cuda or inputs.is_cuda:
            targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return (-targets * log_probs).sum(dim=1).mean()


class IndexedDataset(Dataset):
    """返回 (x, y, idx) 的包装，供目标适应 mem_label 索引（官方 ImageList_idx）。"""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, i


def train_source(enc, clf, src_loader, device, epochs, lr):
    """SHOT 官方源训练：label smoothing + 分层 lr（netF 0.1×, netB/netC 1×）。"""
    optimizer = optim.SGD([{'params': enc.parameters(), 'lr': lr * 0.1},
                           {'params': clf.bottleneck.parameters(), 'lr': lr},
                           {'params': clf.head.parameters(), 'lr': lr}])
    optimizer = op_copy(optimizer)
    criterion = CrossEntropyLabelSmooth(num_classes=clf.head.out_features, epsilon=0.1)
    max_iter = epochs * len(src_loader)
    iter_num = 0
    it = iter(src_loader)
    while iter_num < max_iter:
        try:
            xs, ys = next(it)
        except StopIteration:
            it = iter(src_loader)
            xs, ys = next(it)
        if xs.size(0) == 1:
            continue
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()
        optimizer.zero_grad()
        out = clf(enc(xs))
        loss = criterion(out, ys)
        loss.backward()
        optimizer.step()
    print("  [SHOT] source training done ({} iters)".format(iter_num))


def obtain_label(dataset, enc, clf, device, batch_size=64, distance='cosine', threshold=0):
    """SHOT 官方聚类伪标签（image_target.py obtain_label）：softmax 加权类中心 + cdist 2 轮。

    dataset：目标域训练集（IndexedDataset 包装过）；内部用不 shuffle 的 loader，
    保证返回的伪标签顺序与数据集索引一致（mem_label[tar_idx] 对齐）。
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_fea, all_output, all_label = [], [], []
    enc.eval(); clf.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            f = enc(inputs)
            all_fea.append(clf.forward_features(f).cpu())
            all_output.append(clf(f).cpu())
            all_label.append(labels)
    all_fea = torch.cat(all_fea)
    all_output = torch.cat(all_output)
    all_label = torch.cat(all_label)
    all_output = F.softmax(all_output, dim=1)
    _, predict = torch.max(all_output, 1)
    if distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().numpy()
    K = all_output.size(1)
    aff = all_output.float().numpy()
    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > threshold)[0]
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]
        aff = np.eye(K)[predict]
    return predict.astype('int')


def adapt_target(enc, clf, tgt_train, tgt_test, device, epochs, lr,
                 cls_par=0.3, ent_par=1.0, gent=True, interval=15):
    """SHOT 官方目标适应：聚类伪标签 CE + 熵最小化（netC 冻结）。"""
    for p in clf.head.parameters():
        p.requires_grad_(False)
    optimizer = optim.SGD([{'params': enc.parameters(), 'lr': lr * 0.1},
                           {'params': clf.bottleneck.parameters(), 'lr': lr}])
    optimizer = op_copy(optimizer)
    tgt_train_idx = IndexedDataset(tgt_train.dataset)
    from torch.utils.data import DataLoader
    loader = DataLoader(tgt_train_idx, batch_size=tgt_train.batch_size, shuffle=True)
    max_iter = epochs * len(loader)
    interval_iter = max(1, max_iter // interval)
    iter_num = 0
    mem_label = None
    it = iter(loader)
    enc.train(); clf.train()
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(it)
        except StopIteration:
            it = iter(loader)
            inputs_test, _, tar_idx = next(it)
        if inputs_test.size(0) == 1:
            continue
        if iter_num % interval_iter == 0:
            # 官方 transductive 设置 target=test；我们 train≠test，为索引自洽用 tgt_train 生成伪标签
            mem_label = obtain_label(tgt_train.dataset, enc, clf, device)
            mem_label = torch.from_numpy(mem_label.astype('int64')).to(device)
            enc.train(); clf.train()
        inputs_test = inputs_test.to(device)
        tar_idx = tar_idx.to(device)
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        features = clf.forward_features(enc(inputs_test))
        outputs = clf.predict_from_feature(features)
        loss = torch.tensor(0.0).to(device)
        if cls_par > 0:
            pred = mem_label[tar_idx]
            loss += cls_par * F.cross_entropy(outputs, pred)
        if ent_par > 0 or gent:
            softmax_out = F.softmax(outputs, dim=1)
            ent = -(softmax_out * torch.log(softmax_out + 1e-5)).sum(dim=1).mean()
            if gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                ent -= gentropy
            loss += ent_par * ent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            acc, auc, _ = test(enc, clf, tgt_test, len(tgt_test.dataset),
                               num_classes=clf.head.out_features,
                               class_names=tgt_test.dataset.classes
                               if hasattr(tgt_test.dataset, 'classes') else None)
            print("  [SHOT] iter {}/{} tgt_acc={:.4f}".format(iter_num, max_iter, acc))
    return enc, clf


def train(args):
    set_seed(args.seed)
    data = load_task(args.src, args.tgt, input_size=args.input_size,
                     batch_src=args.batch, batch_tgt=args.batch)
    n_cls = len(data['class_names'])
    print("task={}->{}  classes={}".format(
        LABEL_TO_DATASET[args.src], LABEL_TO_DATASET[args.tgt], data['class_names']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc, clf = build_models(n_cls, device)   # bottleneck 'bn' 官方

    print("  [SHOT] stage 1: source training (label smoothing)")
    train_source(enc, clf, data['src_train'], device, epochs=args.src_epochs, lr=args.lr)

    print("  [SHOT] stage 2: target adaptation (pseudo-label + entropy)")
    adapt_target(enc, clf, data['tgt_train'], data['tgt_test'], device,
                 epochs=args.epochs, lr=args.lr, cls_par=args.cls_par,
                 ent_par=args.ent_par, gent=args.gent, interval=args.interval)

    acc, auc, _ = test(enc, clf, data['tgt_test'], len(data['tgt_test'].dataset),
                       num_classes=n_cls, class_names=data['class_names'])
    print("  [SHOT] final tgt_acc={:.4f} tgt_auc={:.4f}".format(acc, auc))


def main():
    parser = argparse.ArgumentParser(description='SHOT baseline for OCT cross-device UDA')
    parser.add_argument('--src', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--tgt', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--src_epochs', type=int, default=20,
                        help='SHOT 官方源训练 max_epoch=20（image_source.py）')
    parser.add_argument('--epochs', type=int, default=15,
                        help='SHOT 官方目标适应 max_epoch=15')
    parser.add_argument('--batch', type=int, default=64,
                        help='SHOT 官方 batch=64（image_source.py）')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='SHOT 官方 lr=1e-2')
    parser.add_argument('--cls_par', type=float, default=0.3,
                        help='SHOT 官方伪标签权重 cls_par=0.3')
    parser.add_argument('--ent_par', type=float, default=1.0,
                        help='SHOT 官方熵权重 ent_par=1.0')
    parser.add_argument('--gent', type=int, default=1,
                        help='SHOT 官方全局熵 gent=True')
    parser.add_argument('--interval', type=int, default=15,
                        help='SHOT 官方伪标签更新 interval=15')
    parser.add_argument('--input_size', type=int, default=224,
                        help='thuml ResNet 官方输入 224')
    args = parser.parse_args()
    args.gent = bool(args.gent)
    train(args)


if __name__ == '__main__':
    main()

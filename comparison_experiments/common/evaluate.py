"""Unified evaluation: acc + per-class recall + macro AUC + full metrics.

Same as the main project eval_utils; all baseline methods use this for fairness.
"""
import time
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix,
)


def test(encoder, classifier, dataloader, dataset_size, num_classes=3,
         class_names=None, show_progress=False):
    """Evaluate acc + macro-ovr AUC + per-class recall. Returns (acc, auc, per_class_recall)."""
    since = time.time()
    acc = 0
    prob_all = torch.Tensor()
    target_all = torch.Tensor()
    pred_all = torch.Tensor()
    iterator = tqdm(dataloader, desc='Evaluating') if show_progress else dataloader

    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        for inputs, labels in iterator:
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            features = encoder(inputs)
            if isinstance(features, (tuple, list)):
                features = features[0]
            outputs = classifier(features)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            _, preds = outputs.max(1)
            acc += preds.eq(labels).sum().item()
            prob_all = torch.cat((prob_all, outputs.softmax(dim=1).cpu()), dim=0)
            target_all = torch.cat((target_all, labels.cpu()), dim=0)
            pred_all = torch.cat((pred_all, preds.cpu()), dim=0)

    avg_acc = float(acc) / max(dataset_size, 1)
    y = target_all.numpy().astype(int)
    p = pred_all.numpy().astype(int)
    auc = float(roc_auc_score(y, prob_all.numpy(), average='macro', multi_class='ovr'))

    # Full metric set (same as main project TEST_METRICS)
    recall_m = float(recall_score(y, p, average='macro', zero_division=0))
    precision_m = float(precision_score(y, p, average='macro', zero_division=0))
    f1_m = float(f1_score(y, p, average='macro', zero_division=0))
    bacc = float(balanced_accuracy_score(y, p))
    cm = confusion_matrix(y, p, labels=list(range(num_classes)))
    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    with np.errstate(divide='ignore', invalid='ignore'):
        spec = np.nan_to_num(tn / (tn + fp))
        rec = np.nan_to_num(tp / (tp + fn))
    specificity_m = float(spec.mean())
    gmean = float(rec.prod() ** (1.0 / num_classes))
    kappa = float(cohen_kappa_score(y, p))
    mcc = float(matthews_corrcoef(y, p))

    print("  test acc={:.4f} | auc={:.4f} | recall={:.4f} | precision={:.4f} | f1={:.4f} | "
          "bacc={:.4f} | specificity={:.4f} | kappa={:.4f} | gmean={:.4f} | mcc={:.4f} | time={:.1f}s".format(
              avg_acc, auc, recall_m, precision_m, f1_m, bacc, specificity_m,
              kappa, gmean, mcc, time.time() - since))

    per_cls = []
    for c in range(num_classes):
        m = y == c
        tot = int(m.sum())
        per_cls.append(float((p[m] == c).sum()) / max(tot, 1) if tot > 0 else float('nan'))
    if class_names:
        parts = " | ".join("{0}:{1:.4f}".format(class_names[i], per_cls[i]) for i in range(num_classes))
        print("  per-class recall: {}".format(parts))
    return avg_acc, auc, per_cls

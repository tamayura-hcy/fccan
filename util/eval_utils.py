"""Evaluation & diagnostics: test() + feature diagnostics + JSON/TXT persistence."""
import os
import time
import json as _json

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, confusion_matrix,
    f1_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
)


def format_per_class_acc(per_class_acc, class_names=None):
    """Per-class recall; nan means the class is absent from the test set."""
    parts = []
    for i, a in enumerate(per_class_acc):
        name = class_names[i] if class_names is not None and i < len(class_names) else "cls{}".format(i)
        if isinstance(a, float) and a != a:
            parts.append("{}:N/A".format(name))
        else:
            parts.append("{}:{:.4f}".format(name, float(a)))
    return " | ".join(parts)


def heatmap(SR, GT, color='YlGn', class_names=None):
    """Confusion-matrix heatmap; returns an empty array if matplotlib fails to load.

    SR: predicted labels, GT: true labels (1-D arrays).
    """
    if class_names is None:
        class_names = ['AMD', 'DME', 'Normal']
    try:
        import matplotlib
        matplotlib.use('Agg')  # GUI-less backend, less memory
        import matplotlib.pyplot as plt
        import seaborn as sns
    except (ImportError, OSError) as e:
        print("Warning: matplotlib load failed ({}), skip heatmap".format(e))
        return np.zeros((1, 1), dtype=np.uint8)

    plt.clf()
    labels = list(range(len(class_names)))
    cm = confusion_matrix(GT, SR, labels=labels)
    total_samples = np.sum(cm)
    if total_samples <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    percentage_matrix = cm / total_samples
    ax = sns.heatmap(percentage_matrix, square=True, annot=True, fmt='.2%',
                     xticklabels=class_names, yticklabels=class_names, cmap=color)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predict label')
    ax.set_ylabel('True label')
    ax.invert_yaxis()
    fig = plt.gcf()
    fig.set_dpi(300)
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return image


def test(encoder, classifier, dataloader_test, dataset_size_test,
         show_progress=True, report_per_class=False, num_classes=None, class_names=None):
    since = time.time()
    acc_test = 0
    cpu_pseudo_label = torch.Tensor()
    cpu_target = torch.Tensor()
    cpu_predicted = torch.Tensor()
    cpu_features = torch.Tensor()  # collect features for diagnostics
    iterator = tqdm(dataloader_test, desc='Evaluating') if show_progress else dataloader_test
    for i, (inputs, labels) in enumerate(iterator):
        encoder.eval()
        classifier.eval()

        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            features = encoder(inputs)[0]
            outputs = classifier(features)
            _, preds = outputs[0].max(1)

            acc_test += preds.eq(labels).sum().item()
            cpu_pseudo_label = torch.cat((cpu_pseudo_label, outputs[0].softmax(dim=1).cpu()), dim=0)
            cpu_target = torch.cat((cpu_target, labels.cpu()), dim=0)
            cpu_predicted = torch.cat((cpu_predicted, preds.cpu()), dim=0)
            cpu_features = torch.cat((cpu_features, features.cpu()), dim=0)
        del inputs, labels, features, preds
        torch.cuda.empty_cache()

    AUC = roc_auc_score(cpu_target, cpu_pseudo_label, average='macro', multi_class='ovr')
    TPR = recall_score(cpu_predicted, cpu_target, average='macro')
    PPV = precision_score(cpu_predicted, cpu_target, average='macro')
    # ── Extended metrics: macro-F1 / Balanced Acc / Specificity / Kappa / G-mean / MCC ──
    _y = cpu_target.numpy().astype(np.int64)
    _p = cpu_predicted.numpy().astype(np.int64)
    F1 = f1_score(_y, _p, average='macro')
    BACC = balanced_accuracy_score(_y, _p)
    _nc = int(max(_y.max(), _p.max())) + 1
    _cmf = confusion_matrix(_y, _p, labels=list(range(_nc)))
    _specs = []
    for _c in range(_nc):
        _tn = _cmf.sum() - _cmf[_c, :].sum() - _cmf[:, _c].sum() + _cmf[_c, _c]
        _fp = _cmf[:, _c].sum() - _cmf[_c, _c]
        _specs.append(_tn / float(_tn + _fp) if (_tn + _fp) > 0 else 1.0)
    SPEC = float(np.mean(_specs))
    KAPPA = cohen_kappa_score(_y, _p)
    GMEAN = float(np.sqrt(max(0.0, float(TPR) * SPEC)))
    MCC = matthews_corrcoef(_y, _p)
    try:
        matrix = heatmap(cpu_predicted, cpu_target)
    except Exception as e:
        print("Warning: heatmap load failed ({}), skip heatmap".format(e))
        matrix = np.zeros((1, 1), dtype=np.uint8)
    val = [AUC, TPR, PPV, matrix, F1, BACC, SPEC, KAPPA, GMEAN, MCC]
    elapsed_time = time.time() - since
    print("Test completed in {:.2f}s".format(elapsed_time))
    avg_acc = float(acc_test) / dataset_size_test
    print("test acc={:.4f} | auc={:.4f} | recall={:.4f} | precision={:.4f} | f1={:.4f} | bacc={:.4f} | specificity={:.4f} | kappa={:.4f} | gmean={:.4f} | mcc={:.4f}".format(
        avg_acc, AUC, TPR, PPV, F1, BACC, SPEC, KAPPA, GMEAN, MCC))
    if report_per_class and cpu_target.numel() > 0:
        y = cpu_target.numpy().astype(np.int64)
        p = cpu_predicted.numpy().astype(np.int64)
        f = cpu_features.numpy().astype(np.float32)  # (N, 2048)
        if num_classes is None:
            num_classes = int(max(y.max(), p.max())) + 1
        per_cls = []
        for c in range(num_classes):
            m = y == c
            tot = int(m.sum())
            if tot == 0:
                per_cls.append(float('nan'))
            else:
                per_cls.append(float((p[m] == c).sum()) / float(tot))
        print("  per-class recall: {}".format(format_per_class_acc(per_cls, class_names)))
        # Confusion matrix
        _cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for _i in range(len(y)):
            _cm[y[_i], p[_i]] += 1
        _names = class_names if class_names else [str(c) for c in range(num_classes)]
        print("  confusion matrix [row=true, col=pred]:")
        _header = "         " + "".join("{:>8s}".format(n) for n in _names)
        print(_header)
        for _c in range(num_classes):
            _row = "  {:>6s} ".format(_names[_c]) + "".join("{:>8d}".format(int(_cm[_c, _j])) for _j in range(num_classes))
            print(_row)
        # Track AMD->NORMAL misclassifications
        _amd2normal = np.where((y == 0) & (p == 2))[0]
        if len(_amd2normal) > 0:
            print("  AMD->NORMAL n={}: {}".format(
                len(_amd2normal), sorted(_amd2normal.tolist())))
        _normal2dme = np.where((y == 2) & (p == 1))[0]
        if len(_normal2dme) > 0:
            print("  NORMAL->DME n={}: {}".format(
                len(_normal2dme), sorted(_normal2dme.tolist())))

        # ====== Additional diagnostics ======
        diag_dict = compute_feature_diagnostics(f, y, p, num_classes, _names)
        # Classifier weight norms: verify the "high-norm DME pulls in NORMAL" hypothesis
        _cls_w_norms = get_classifier_weight_norms(classifier)
        if _cls_w_norms is not None:
            diag_dict['cls_w_norms'] = _cls_w_norms
        print_diagnostic_report(diag_dict, _names)

        # Write JSON log for diagnostic scripts
        try:
            write_diag_json(diag_dict, avg_acc, per_cls, _cm, _amd2normal, _normal2dme, _names)
        except Exception as _e:
            print(f"  [WARN] diag JSON write failed: {_e}")
        # Append to diagnosis_results.txt right away (saved every epoch)
        append_diag_txt(avg_acc, per_cls, _amd2normal, _normal2dme, diag_dict, _names)

    print()
    torch.cuda.empty_cache()
    return avg_acc, val


def get_classifier_weight_norms(classifier):
    """L2 norm ||w_c|| of each class weight in the last classifier layer (norm-attraction hypothesis)."""
    try:
        # classifier structure: 2240->1024->1024->3
        for name, module in classifier.named_modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == 3:
                w = module.weight.detach().cpu().numpy()  # [3, 1024]
                return [float(np.linalg.norm(w[i])) for i in range(w.shape[0])]
        return None
    except Exception:
        return None


def compute_feature_diagnostics(features, y_true, y_pred, num_classes, class_names):
    """Compute feature diagnostics; return a dict."""
    eps = 1e-8
    f = features.astype(np.float64)
    diag = {'num_classes': int(num_classes)}

    # 1. class-prototype cosine similarity
    prototypes = np.zeros((num_classes, f.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            prototypes[c] = f[mask].mean(axis=0)
    proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + eps)
    cos_mat = proto_norm @ proto_norm.T
    diag['proto_cos'] = [[float(cos_mat[i, j]) for j in range(num_classes)] for i in range(num_classes)]
    diag['proto_warnings'] = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if cos_mat[i, j] > 0.5:
                diag['proto_warnings'].append(f"{class_names[i]}↔{class_names[j]}={cos_mat[i, j]:.4f}")

    # 2. feature-dimension collapse
    try:
        f_centered = f - f.mean(axis=0, keepdims=True)
        cov = (f_centered.T @ f_centered) / (f.shape[0] - 1)
        sv = np.linalg.svd(cov, compute_uv=False)
        total = sv.sum()
        cumsum = np.cumsum(sv) / total
        rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)
        diag['eff_rank'] = rank_90
        diag['collapse_ratio'] = float(rank_90 / f.shape[1])
    except Exception:
        diag['eff_rank'] = -1
        diag['collapse_ratio'] = -1.0

    # 3. per-class feature norms
    norms_dict = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            n = np.linalg.norm(f[mask], axis=1)
            norms_dict[class_names[c]] = [float(n.mean()), float(n.std()), float(n.min()), float(n.max())]
    diag['feature_norms'] = norms_dict

    # 4. effective rank of the orthogonal subspace (sensitivity beyond total effective rank)
    try:
        # Orthonormal basis spanning the prototypes (Gram-Schmidt)
        basis = []
        for i in range(num_classes):
            v = proto_norm[i].copy()
            for b in basis:
                v = v - np.dot(v, b) * b
            vn = np.linalg.norm(v)
            if vn > 1e-6:
                basis.append(v / vn)
        if len(basis) > 0:
            B = np.stack(basis)  # [k, D]
            f_orth = f - (f @ B.T) @ B  # [N, D] orthogonal-subspace features
            f_orth_c = f_orth - f_orth.mean(axis=0, keepdims=True)
            s_orth = np.linalg.svd(f_orth_c, compute_uv=False)
            p_orth = s_orth / (s_orth.sum() + eps)
            orth_rank = float(np.exp(-np.sum(p_orth * np.log(p_orth + eps))))
            diag['orth_eff_rank'] = orth_rank
            diag['proto_basis_rank'] = len(basis)
        else:
            diag['orth_eff_rank'] = -1.0
            diag['proto_basis_rank'] = 0
    except Exception:
        diag['orth_eff_rank'] = -1.0
        diag['proto_basis_rank'] = -1

    return diag


def print_diagnostic_report(diag, class_names):
    """Print the diagnostic report."""
    # Prototype cosines
    print("  prototype cosine [1=confusable, >0.8=severe overlap]:")
    _header = "         " + "".join("{:>8s}".format(n) for n in class_names)
    print(_header)
    cm = diag['proto_cos']
    for _c in range(diag['num_classes']):
        _row = "  {:>6s} ".format(class_names[_c]) + "".join("{:8.4f}".format(float(cm[_c][_j])) for _j in range(diag['num_classes']))
        print(_row)
    for w in diag.get('proto_warnings', []):
        print(f"  WARN {w} (cos>0.5, feature overlap)")

    # Dimension collapse
    er = diag.get('eff_rank', -1)
    cr = diag.get('collapse_ratio', -1.0)
    print(f"  eff_rank={er}/2048 ({cr:.1%})")


def write_diag_json(diag, avg_acc, per_cls, cm, a2n, n2d, class_names):
    """Append the JSON log under the saves dir."""
    save_dir = "saves/DAGCN_BOE_to_TMI_iter1"
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "diag_log.jsonl")
    record = {
        'acc': float(avg_acc),
        'per_class': {class_names[c]: float(per_cls[c]) if c < len(per_cls) else float('nan') for c in range(diag['num_classes'])},
        'cm': [[int(cm[i, j]) for j in range(diag['num_classes'])] for i in range(diag['num_classes'])],
        'a2n_count': int(len(a2n)),
        'n2d_count': int(len(n2d)),
        'proto_cos': diag.get('proto_cos', []),
        'proto_warnings': diag.get('proto_warnings', []),
        'eff_rank': diag.get('eff_rank', -1),
        'collapse_ratio': diag.get('collapse_ratio', -1.0),
        'feature_norms': diag.get('feature_norms', {}),
        'orth_eff_rank': diag.get('orth_eff_rank', -1),
        'proto_basis_rank': diag.get('proto_basis_rank', -1),
        'cls_w_norms': diag.get('cls_w_norms'),
    }
    with open(json_path, 'a', encoding='utf-8') as f:
        f.write(_json.dumps(record, ensure_ascii=False) + '\n')
    # epoch counter
    if not hasattr(write_diag_json, '_count'):
        write_diag_json._count = 0
    write_diag_json._count += 1
    ep = write_diag_json._count
    print(f"  diag saved ep{ep} -> {json_path}")


def append_diag_txt(avg_acc, per_cls, a2n, n2d, diag, class_names):
    """Append one line per epoch to diagnosis_results.txt."""
    txt_path = "diagnosis_results.txt"
    pc = {class_names[c]: per_cls[c] if c < len(per_cls) else float('nan') for c in range(diag['num_classes'])}
    er = diag.get('eff_rank', -1)
    cr = diag.get('collapse_ratio', -1.0)
    cos = diag.get('proto_cos', [])
    fn = diag.get('feature_norms', {})
    pw = ';'.join(diag.get('proto_warnings', []))

    # Prototype-cosine matrix summary
    cos_str = ""
    if cos:
        off_diag = []
        for i in range(len(cos)):
            for j in range(i + 1, len(cos)):
                off_diag.append(f"{class_names[i][:3]}-{class_names[j][:3]}={cos[i][j]:.3f}")
        cos_str = " cos[" + " ".join(off_diag) + "]"

    # Feature-norm summary
    norm_str = ""
    if fn:
        parts = [f"{k[:3]}={fn[k][0]:.1f}" for k in fn]
        norm_str = " norm[" + " ".join(parts) + "]"

    # Orthogonal-subspace effective rank + classifier weight norms
    oer = diag.get('orth_eff_rank', -1)
    pbr = diag.get('proto_basis_rank', -1)
    cwn = diag.get('cls_w_norms')
    orth_str = f" orthR={oer:.0f}(b={pbr})" if oer > 0 else ""
    cwn_str = ""
    if cwn and len(cwn) == diag['num_classes']:
        cwn_str = " ||w||[" + " ".join(f"{class_names[c][:3]}={cwn[c]:.1f}" for c in range(diag['num_classes'])) + "]"

    line = (f"acc={avg_acc:.4f} AMD={pc.get('AMD', 0):.3f} DME={pc.get('DME', 0):.3f} N={pc.get('NORMAL', 0):.3f} "
            f"A2N={len(a2n)} N2D={len(n2d)} rank={er}({cr * 100:.0f}%){cos_str}{norm_str}{orth_str}{cwn_str}")
    if pw:
        line += f" [W]{pw}"
    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        f.flush()

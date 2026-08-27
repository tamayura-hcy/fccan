"""Unified diagnostics: concise one-line output + JSON persistence.

All metrics are scalars or fixed-length arrays for easy plotting/comparison.

Key groups: E0 confusion/recall, E3 prototype cosine, E4 feature norms, E5 classifier
weight norms, A1 logit stats, A2 decision-boundary distance, A3 ECE, B-FLIP boundary
flip audit, B1 Fisher ratio, module diagnostics (energy gap / ETF deviation / CaCo Q).
"""

import torch
import torch.nn.functional as F
import numpy as np


# ════════════════════════════════════════════════════════════
#  E0. Confusion matrix + per-class recall + key misclassification indices
# ════════════════════════════════════════════════════════════

def confusion_metrics(labels, preds, num_classes, class_names=None):
    C = num_classes
    cm = np.zeros((C, C), dtype=np.int64)
    y = labels.numpy().astype(np.int64)
    p = preds.numpy().astype(np.int64)
    for i in range(len(y)):
        cm[y[i], p[i]] += 1
    recall = []
    precision = []
    for c in range(C):
        tot = cm[c].sum()
        recall.append(round(cm[c, c] / max(tot, 1), 4))
        col_tot = cm[:, c].sum()
        precision.append(round(cm[c, c] / max(col_tot, 1), 4))
    f1s = [round(2 * r * p / max(r + p, 1e-8), 4) for r, p in zip(recall, precision)]
    macro_f1 = round(float(np.mean(f1s)), 4)
    offenders = {}
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            mask = (y == i) & (p == j)
            idxs = np.where(mask)[0].tolist()
            if idxs:
                key = f"{class_names[i]}->{class_names[j]}" if class_names else f"{i}->{j}"
                offenders[key] = idxs[:50]
    return dict(cm=cm.tolist(), recall=recall, precision=precision, f1=f1s, macro_f1=macro_f1, offenders=offenders)


# ════════════════════════════════════════════════════════════
#  E3. Prototype cosine similarity matrix
# ════════════════════════════════════════════════════════════

def prototype_cosine_matrix(feats, labels, num_classes):
    C = num_classes
    feats_n = F.normalize(feats.float(), dim=1, eps=1e-8)
    protos = []
    for c in range(C):
        m = (labels == c)
        if m.any():
            protos.append(feats_n[m].mean(dim=0))
        else:
            protos.append(torch.zeros(feats.size(1)))
    protos = F.normalize(torch.stack(protos, dim=0), dim=1, eps=1e-8)
    cos_mat = protos @ protos.t()
    return cos_mat.numpy()


# ════════════════════════════════════════════════════════════
#  E4. Per-class feature L2 norms
# ════════════════════════════════════════════════════════════

def per_class_feature_norm(feats, labels, num_classes):
    C = num_classes
    norms = []
    for c in range(C):
        m = (labels == c)
        if m.any():
            n = feats[m].norm(dim=1).float()
            norms.append(dict(
                mean=round(n.mean().item(), 2), std=round(n.std().item(), 2),
                min=round(n.min().item(), 2), max=round(n.max().item(), 2)))
        else:
            norms.append(dict(mean=0, std=0, min=0, max=0))
    return norms


# ════════════════════════════════════════════════════════════
#  E5. Classifier weight norms
# ════════════════════════════════════════════════════════════

def _get_last_linear_weight(classifier):
    """Extract the last Linear layer's weight from any classifier."""
    if hasattr(classifier, 'weight') and isinstance(classifier.weight, torch.Tensor):
        return classifier.weight
    # Classifier(classifier=Sequential(...)) pattern
    if hasattr(classifier, 'classifier'):
        inner = classifier.classifier
        if hasattr(inner, 'weight') and isinstance(inner.weight, torch.Tensor):
            return inner.weight
        if isinstance(inner, torch.nn.Sequential):
            for m in reversed(list(inner.children())):
                if isinstance(m, torch.nn.Linear):
                    return m.weight
    # Walk all submodules to find the last Linear
    last_linear = None
    for m in classifier.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    if last_linear is not None:
        return last_linear.weight
    raise AttributeError('Cannot find a Linear layer in classifier to extract weight')


def classifier_weight_norms(classifier):
    W = _get_last_linear_weight(classifier).detach().float()
    norms = W.norm(dim=1).tolist()
    return [round(n, 3) for n in norms]


# ════════════════════════════════════════════════════════════
#  A1. Logit-distribution drift monitor
# ════════════════════════════════════════════════════════════

def logit_distribution_stats(logits, labels, num_classes):
    C = num_classes
    stats = {}
    for c in range(C):
        mask = (labels == c)
        if not mask.any():
            stats[c] = dict(mean=float('nan'))
            continue
        cls_logits = logits[mask, c].float()
        m_val = cls_logits.mean().item()
        s_val = cls_logits.std().item()
        if s_val > 1e-8:
            z = (cls_logits - m_val) / s_val
            skew = (z ** 3).mean().item()
        else:
            skew = 0.0
        q = torch.quantile(cls_logits, torch.tensor([0.25, 0.5, 0.75], device=cls_logits.device))
        stats[c] = dict(mean=round(m_val, 3), std=round(s_val, 3), skew=round(skew, 3),
                        q25=round(q[0].item(), 3), q50=round(q[1].item(), 3), q75=round(q[2].item(), 3))
    return stats


# ════════════════════════════════════════════════════════════
#  A2. Decision-boundary distance distribution (logits-based, any classifier structure)
# ════════════════════════════════════════════════════════════

def decision_boundary_distances(logits, labels=None, preds=None):
    """d_i = logit_yhat - max_{c != yhat} logit_c; computed from logits only, no weight matrix needed."""
    if preds is None:
        preds = logits.argmax(dim=1)
    d_correct = logits.gather(1, preds.unsqueeze(1)).squeeze(1)  # [N]
    logits_masked = logits.clone()
    logits_masked.scatter_(1, preds.unsqueeze(1), float('-inf'))
    d_max_wrong = logits_masked.max(dim=1)[0]
    d = d_correct - d_max_wrong
    d_np = d.numpy()
    summary = dict(
        mean=round(float(d_np.mean()), 4), std=round(float(d_np.std()), 4),
        q50=round(float(np.percentile(d_np, 50)), 4),
        frac_neg=round(float((d_np < 0).mean()), 4),
        frac_deep_neg=round(float((d_np < -0.3).mean()), 4))
    offender_d = None
    if labels is not None:
        off_mask = (labels != preds)
        if off_mask.any():
            offender_d = d[off_mask]
            summary['n_offenders'] = int(off_mask.sum().item())
            summary['off_d_mean'] = round(float(offender_d.mean().item()), 3)
            summary['off_frac_border'] = round(
                float(((offender_d >= -0.1) & (offender_d <= 0)).float().mean().item()), 3)
            summary['off_frac_deep'] = round(
                float((offender_d < -0.3).float().mean().item()), 3)
    return dict(d=d, summary=summary, offender_d=offender_d)


# ════════════════════════════════════════════════════════════
#  B-FLIP. Boundary-flip audit (measure only, no training)
#
#  For the most entangled class pair (i,j), inspect how i->j errors get flipped:
#    margin = logit_true(i) - logit_pred(j)   (more negative = flipped harder)
#    |margin| < border  -> near-boundary error (boundary compensation can fix)
#    margin < -deep     -> deep error (feature issue, boundary compensation cannot fix)
#  Output: pair, nFlip, margMean, fracBorder, fracDeep.
# ════════════════════════════════════════════════════════════

def boundary_flip_audit(logits, labels, num_classes, class_names=None,
                        border=0.5, deep=2.0):
    C = num_classes
    cn = class_names or [f"C{c}" for c in range(C)]
    if not torch.is_tensor(logits):
        return None
    preds = logits.argmax(dim=1)
    y = labels
    # Find the most entangled error pair (i != j with most errors)
    best = None
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            m = (y == i) & (preds == j)
            n = int(m.sum().item())
            if best is None or n > best[2]:
                best = (i, j, n, m)
    if best is None or best[2] == 0:
        return None
    i, j, n, m = best
    li = logits[m, i].float()
    lj = logits[m, j].float()
    margin = (li - lj)  # true-class logit - wrong-class logit, <0
    mm = margin
    frac_border = float(((mm.abs() < border)).float().mean().item())
    frac_deep = float((mm < -deep).float().mean().item())
    return {
        'pair': f"{cn[i]}->{cn[j]}",
        'n_flip': n,
        'marg_mean': round(float(mm.mean().item()), 3),
        'frac_border': round(frac_border, 3),
        'frac_deep': round(frac_deep, 3),
    }


def intra_class_cohesion_probe(feats, logits, labels, num_classes,
                               class_names=None):
    """Intra-class cohesion probe (checks the CaCo key self-cleaning premise).

    For each sample: mean cosine vs other samples in its predicted class -> cohesion.
    Misclassified samples should be isolated in their pseudo class, so they can be
    down-weighted as CaCo keys; reports err vs correct cohesion + the most entangled pair.
    """
    C = num_classes
    cn = class_names or [f"C{c}" for c in range(C)]
    if not torch.is_tensor(feats) or not torch.is_tensor(logits):
        return None
    feats_n = F.normalize(feats.float(), dim=1, eps=1e-8)
    preds = logits.argmax(dim=1)
    sim = feats_n @ feats_n.T                      # [N, N] pairwise cosine
    N = feats_n.size(0)
    cohesion = torch.zeros(N, device=feats_n.device)
    for c in range(C):
        m = (preds == c)
        nc = int(m.sum().item())
        if nc < 2:
            cohesion[m] = float('nan')
            continue
        s = sim[m][:, m].clone()                   # within-class pairwise similarity
        s.fill_diagonal_(0.0)                      # drop self
        cohesion[m] = s.sum(dim=1) / (nc - 1)      # mean cosine vs same-class samples
    err_m = (preds != labels)
    cor_m = ~err_m
    err_c = cohesion[err_m]
    cor_c = cohesion[cor_m]
    out = {
        'err_n': int(err_m.sum().item()),
        'err_mean': round(float(err_c.mean().item()), 4) if err_m.any() else float('nan'),
        'err_med': round(float(err_c.median().item()), 4) if err_m.any() else float('nan'),
        'cor_mean': round(float(cor_c.mean().item()), 4) if cor_m.any() else float('nan'),
        'cor_med': round(float(cor_c.median().item()), 4) if cor_m.any() else float('nan'),
    }
    # Gap criterion: misclassified cohesion should be clearly lower (larger gap is better)
    if err_m.any() and cor_m.any():
        out['gap'] = round(out['cor_mean'] - out['err_mean'], 4)
    else:
        out['gap'] = float('nan')
    # Inspect the most entangled error pair separately (e.g. AMD->NORMAL mixed into NORMAL)
    best = None
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            m = (labels == i) & (preds == j)
            n = int(m.sum().item())
            if best is None or n > best[2]:
                best = (i, j, n, m)
    if best is not None and best[2] > 0:
        i, j, n, m = best
        # Control group: true class j (labels==j & preds==j) cohesion
        true_j = (labels == j) & (preds == j)
        out['pair'] = f"{cn[i]}->{cn[j]}"
        out['pair_n'] = n
        out['pair_mean'] = round(float(cohesion[m].mean().item()), 4)
        out['true_j_mean'] = round(float(cohesion[true_j].mean().item()), 4) if true_j.any() else float('nan')
        out['pair_gap'] = round(out['true_j_mean'] - out['pair_mean'], 4) if true_j.any() else float('nan')
    return out


# ════════════════════════════════════════════════════════════
#  A3. Confidence calibration curve (ECE)
# ════════════════════════════════════════════════════════════

def per_class_conf_entropy(logits, labels, num_classes):
    """Per-true-class mean top-1 conf and mean entropy (leading ep2->ep3 collapse signal)."""
    probs = F.softmax(logits.float(), dim=1)
    conf, _ = probs.max(dim=1)
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    out = {}
    for c in range(num_classes):
        m = (labels == c)
        if not m.any():
            out[c] = dict(conf=float('nan'), ent=float('nan'))
            continue
        out[c] = dict(conf=round(float(conf[m].mean().item()), 4),
                      ent=round(float(ent[m].mean().item()), 4))
    return out


def free_energy_by_class(logits, labels, num_classes, tau=1.0):
    """Per-true-class mean free energy F=-tau*logsumexp(logits/tau) + overall mean. More negative = more confident."""
    lg = logits.float()
    Fv = -float(tau) * torch.logsumexp(lg / float(tau), dim=1)
    out = {}
    for c in range(num_classes):
        m = (labels == c)
        out[c] = round(float(Fv[m].mean().item()), 4) if m.any() else float('nan')
    out['all'] = round(float(Fv.mean().item()), 4)
    return out


def expected_calibration_error(logits, labels, n_bins=10):
    probs = F.softmax(logits.float(), dim=1)
    conf, preds = probs.max(dim=1)
    correct = (preds == labels).float()
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece, N = 0.0, logits.size(0)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        n_b = mask.sum().item()
        if n_b == 0:
            continue
        ece += (n_b / N) * abs(conf[mask].mean().item() - correct[mask].mean().item())
    return round(ece, 4)


# ════════════════════════════════════════════════════════════
#  B1. Fisher discriminant ratio
# ════════════════════════════════════════════════════════════

def fisher_discriminant_ratio(feats, labels, num_classes):
    C = num_classes
    centroids = []
    for c in range(C):
        m = (labels == c)
        centroids.append(feats[m].mean(dim=0) if m.any() else torch.zeros(feats.size(1)))
    centroids = torch.stack(centroids, dim=0)
    intra_stds = []
    for c in range(C):
        m = (labels == c)
        if m.sum() > 1:
            intra_stds.append(feats[m].std(dim=0).mean().item())
        else:
            intra_stds.append(0.0)
    inter_dists = []
    for i in range(C):
        for j in range(i + 1, C):
            inter_dists.append((centroids[i] - centroids[j]).norm().item())
    mean_intra = np.mean(intra_stds)
    mean_inter = np.mean(inter_dists) if inter_dists else 1.0
    fisher = mean_inter / max(mean_intra, 1e-8)
    return dict(fisher=round(fisher, 2), intra=round(mean_intra, 4), inter=round(mean_inter, 4))


# ════════════════════════════════════════════════════════════
#  B2. Prototype-sample cosine histogram
# ════════════════════════════════════════════════════════════

def proto_sample_cosine_histogram(feats, labels, num_classes, n_bins=10):
    C = num_classes
    feats_n = F.normalize(feats.float(), dim=1, eps=1e-8)
    centroids = []
    for c in range(C):
        m = (labels == c)
        centroids.append(feats_n[m].mean(dim=0) if m.any() else torch.zeros(feats.size(1)))
    centroids = F.normalize(torch.stack(centroids, dim=0), dim=1, eps=1e-8)
    cos_vals = feats_n @ centroids.t()
    low_cos = {}
    for c in range(C):
        m = (labels == c)
        if m.any():
            own = cos_vals[m, c].numpy()
            low_cos[c] = round(float((own < 0.5).mean()), 3)
    return dict(low_cos_frac=low_cos)


# ════════════════════════════════════════════════════════════
#  C1. Loss gradient-conflict matrix
# ════════════════════════════════════════════════════════════

def loss_gradient_conflict(loss_dict, shared_params, retain_graph=True):
    names = list(loss_dict.keys())
    grads, grad_norms = {}, {}
    for name, loss in loss_dict.items():
        if not torch.is_tensor(loss) or loss.ndim != 0:
            continue
        g = torch.autograd.grad(loss, shared_params, retain_graph=retain_graph, allow_unused=True)
        g_flat = torch.cat([x.view(-1) if x is not None else torch.zeros(0, device=loss.device) for x in g])
        grads[name] = g_flat
        grad_norms[name] = round(g_flat.norm().item(), 4)
    pairwise = {}
    for i, ni in enumerate(names):
        if ni not in grads:
            continue
        for j, nj in enumerate(names):
            if nj not in grads or j <= i:
                continue
            cos = F.cosine_similarity(grads[ni].unsqueeze(0), grads[nj].unsqueeze(0)).item()
            pairwise[f"{ni}__vs__{nj}"] = round(cos, 3)
    return dict(pairwise=pairwise, norms=grad_norms)


# ════════════════════════════════════════════════════════════
#  D0. [removed] band diagnostics (freq_band_class_domain_diag / transfer / combo)
#      removed 2026-08-01: the two-day frequency exploration is over
# ════════════════════════════════════════════════════════════


def _fisher_ratio(feats, labels):
    """Between-class variance / within-class variance; larger = better class separation."""
    C = int(labels.max().item()) + 1
    centroids = []
    intra = []
    for c in range(C):
        m = (labels == c)
        if m.sum() < 2: continue
        f_c = feats[m].float()
        centroids.append(f_c.mean(0))
        intra.append(f_c.std(0).mean().item())
    if len(centroids) < 2: return 0.0
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            inter_dists.append((centroids[i]-centroids[j]).norm().item())
    mean_intra = float(np.mean(intra)) if intra else 1.0
    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    return mean_inter / max(mean_intra, 1e-8)


def _domain_separability(feats_src, feats_tgt):
    """Linear-SVM domain classification acc - 0.5; larger = domains more separable."""
    from sklearn.svm import LinearSVC
    N_s = feats_src.size(0)
    N_t = feats_tgt.size(0)
    if min(N_s, N_t) < 5: return 0.0
    X = torch.cat([feats_src, feats_tgt], dim=0).cpu().numpy()
    y = np.array([0]*N_s + [1]*N_t)
    clf = LinearSVC(max_iter=1000, dual=False, random_state=42)
    clf.fit(X, y)
    acc = clf.score(X, y)
    return round(acc - 0.5, 4)  # subtract the random baseline 0.5


# ════════════════════════════════════════════════════════════
#  Combined collector
# ════════════════════════════════════════════════════════════

class DiagV2Collector:
    def __init__(self, num_classes=3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"C{c}" for c in range(num_classes)]
        self.history = []
        self._prev_protos = None
        self._energy_state = None
        self._dual_proto_bank = None

    def set_energy_state(self, energy_state):
        """Inject EnergyUdaState for the eaGap / sconGap diagnostics."""
        self._energy_state = energy_state

    def set_dual_proto_bank(self, dual_proto_bank):
        """Inject DualPrototypeBank for the orthDeg diagnostic."""
        self._dual_proto_bank = dual_proto_bank

    def compute_epoch_diag(self, feats, logits, labels, preds=None,
                           classifier=None):
        if preds is None:
            preds = logits.argmax(dim=1)
        d = {}
        C = self.num_classes
        cn = self.class_names

        # ── Classification metrics ──
        d['E0_conf'] = confusion_metrics(labels, preds, C, cn)
        protos = self._compute_prototypes(feats, labels)
        # proto_drift: prototype L2 drift vs the previous epoch (convergence measure)
        if self._prev_protos is not None and protos.shape == self._prev_protos.shape:
            d['proto_drift'] = round(float((protos - self._prev_protos).norm().item()
                                            / max(protos.norm().item(), 1e-8)), 4)
        self._prev_protos = protos.detach().clone()
        d['E3_proto_cos'] = np.round(prototype_cosine_matrix(feats, labels, C), 4).tolist()
        d['E4_fnorm'] = per_class_feature_norm(feats, labels, C)
        if classifier is not None:
            d['E5_wnorm'] = classifier_weight_norms(classifier)

        # ── Distribution / calibration metrics ──
        d['A1_logit'] = logit_distribution_stats(logits, labels, C)
        a2 = decision_boundary_distances(logits, labels=labels, preds=preds)
        d['A2_boundary'] = a2['summary']
        d['A3_ece'] = expected_calibration_error(logits, labels)
        d['B1_fisher'] = fisher_discriminant_ratio(feats, labels, C)

        # ── Energy UDA diagnostics ──
        #    eaGap: src/tgt free-energy EMA gap -> SCAL alignment (0 is good)
        #    sconGap: max abs per-class energy EMA gap -> SCON normalization (0 is good)
        d['energy_gap'] = None
        d['scon_gap'] = None
        if self._energy_state is not None:
            mu_s = float(self._energy_state.mu_fe_s.item())
            mu_t = float(self._energy_state.mu_fe_t.item())
            d['energy_gap'] = round(mu_t - mu_s, 4)
            # Max per-class energy gap |mu_e_t - mu_e_s|
            per_cls_gap = (self._energy_state.mu_e_t - self._energy_state.mu_e_s).abs()
            d['scon_gap'] = round(float(per_cls_gap.max().item()), 4)

        # ── Norm floor: min/max norm ratio (<0.85 means the weak class is suppressed)
        fnorm_vals = [f["mean"] for f in d.get('E4_fnorm', []) if f.get("mean", 0) > 0]
        if fnorm_vals:
            d['norm_ratio'] = round(min(fnorm_vals) / max(fnorm_vals), 4)
        else:
            d['norm_ratio'] = None

        # ── ETF structural deviation: actual prototype cos vs ideal ETF cos(-1/(K-1)), lower is better
        pcos_mat = d.get('E3_proto_cos')
        if pcos_mat is not None:
            K = len(pcos_mat)
            etf_target = -1.0 / max(K - 1, 1)
            etf_dev = 0.0
            cnt = 0
            for i in range(K):
                for j in range(K):
                    if i != j:
                        etf_dev += abs(float(pcos_mat[i][j]) - etf_target)
                        cnt += 1
            d['etf_dev'] = round(etf_dev / max(cnt, 1), 4)
        else:
            d['etf_dev'] = None

        # ── Dual-proto orthogonality: ||P @ D^T||_F^2 (0 is good; class and domain prototypes orthogonal)
        d['orth_deg'] = None
        if self._dual_proto_bank is not None:
            _p = self._dual_proto_bank.get_proto()
            _ds, _dt = self._dual_proto_bank.get_domain_proto()
            if _p is not None and _ds is not None and _dt is not None:
                _p_n = F.normalize(_p.detach(), dim=1, eps=1e-8)
                _D = torch.stack([_ds.detach(), _dt.detach()], dim=0)
                _D_n = F.normalize(_D, dim=1, eps=1e-8)
                _PD = _p_n @ _D_n.T  # [K, 2]
                d['orth_deg'] = round(float((_PD ** 2).sum().item()), 4)

        # ── CaCo contrast quality: mean inter-class cos - mean intra-class cos (larger is better)
        d['caco_q'] = None
        if protos is not None and feats is not None:
            feat_n = F.normalize(feats, dim=1, eps=1e-8)
            cos_all = feat_n @ protos.T  # [N, K]
            intra_vals, inter_vals = [], []
            for c in range(C):
                mc = (labels == c)
                if mc.any():
                    intra_vals.append(cos_all[mc, c].mean().item())
                    for c2 in range(C):
                        if c2 != c:
                            inter_vals.append(cos_all[mc, c2].mean().item())
            if intra_vals and inter_vals:
                d['caco_q'] = round(
                    sum(intra_vals) / len(intra_vals) - sum(inter_vals) / len(inter_vals), 4)

        # ── Boundary-flip audit: how the most entangled pair's errors get flipped (measure only)
        d['bflip'] = boundary_flip_audit(logits, labels, C, cn)

        # ── Intra-class cohesion probe: are misclassified samples isolated in their pseudo class?
        d['cohesion'] = intra_class_cohesion_probe(feats, logits, labels, C, cn)


        # ── ep1 divergence leading signals (scalarized for plotting) ──
        # confT: per-class mean confidence
        d['D1_conf_ent'] = per_class_conf_entropy(logits, labels, C)

        self.history.append(d)
        return d

    def _compute_prototypes(self, feats, labels):
        C = self.num_classes
        protos = []
        for c in range(C):
            m = (labels == c)
            protos.append(feats[m].mean(dim=0) if m.any() else torch.zeros(feats.size(1)))
        return torch.stack(protos, dim=0)

    def format_epoch_line(self, epoch_diag, epoch_num):
        """One compact line: DIAG epN | key=val | ..."""
        cn = self.class_names
        p = [f"DIAG ep{epoch_num}"]

        # Recall
        conf = epoch_diag.get('E0_conf', {})
        rec = conf.get('recall', [])
        if rec:
            p.append("rec=" + ",".join(f"{r:.4f}" for r in rec))

        # Proto cos max off-diag
        pcos = epoch_diag.get('E3_proto_cos', None)
        if pcos is not None:
            max_off = 0.0
            for i in range(len(pcos)):
                for j in range(len(pcos)):
                    if i != j:
                        max_off = max(max_off, float(pcos[i][j]))
            p.append(f"pcos_max={max_off:.3f}")

        # F norm
        fnorms = epoch_diag.get('E4_fnorm', [])
        if fnorms:
            means = [f["mean"] for f in fnorms]
            p.append("fnorm=" + "/".join(f"{m:.1f}" for m in means))

        # W norm
        wnorm = epoch_diag.get('E5_wnorm', None)
        if wnorm is not None:
            p.append("wnorm=" + "/".join(f"{w:.3f}" for w in wnorm))

        # A1 logit means
        a1 = epoch_diag.get('A1_logit', {})
        if a1:
            lmeans = []
            for c in range(self.num_classes):
                s = a1.get(c, {})
                if not np.isnan(s.get('mean', float('nan'))):
                    lmeans.append(f"{cn[c]}={s['mean']:.2f}")
            if lmeans:
                p.append("logit(" + " ".join(lmeans) + ")")

        # ── Module diagnostics ──
        # Angular repulsion (ETF)
        ed = epoch_diag.get('etf_dev', None)
        if ed is not None:
            p.append(f"etfDev={ed:.3f}")

        # CaCo contrast quality
        cq = epoch_diag.get('caco_q', None)
        if cq is not None:
            p.append(f"cacoQ={cq:+.3f}")

        # proto_drift: prototype L2 drift vs the previous epoch (convergence measure)
        pd = epoch_diag.get('proto_drift', None)
        if pd is not None:
            p.append(f"pdrift={pd:.4f}")

        # Intra-class cohesion probe: err vs correct (larger gap > 0, stronger premise)
        coh = epoch_diag.get('cohesion', None)
        if coh is not None:
            em = coh.get('err_mean', float('nan'))
            cm = coh.get('cor_mean', float('nan'))
            g = coh.get('gap', float('nan'))
            s = f"coh(err={em:.3f},cor={cm:.3f},gap={g:+.3f}"
            if 'pair' in coh:
                pg = coh.get('pair_gap', float('nan'))
                s += f",{coh['pair']}:{pg:+.3f}"
            s += ")"
            p.append(s)

        # Boundary-flip audit: how the most entangled pair's errors get flipped (measure only)
        bf = epoch_diag.get('bflip', None)
        if bf is not None:
            p.append(
                f"bflip[{bf.get('pair','?')}] n={bf.get('n_flip',0)}"
                f" marg={bf.get('marg_mean',0):.2f}"
                f" bord={bf.get('frac_border',0):.2f}"
                f" deep={bf.get('frac_deep',0):.2f}")

        # F^2DP band energy: mean L2 norms of the 4 subbands after DWT
        freq_nrg = epoch_diag.get('freq_nrg', None)
        if freq_nrg is not None:
            p.append(
                f"nrg(LL={freq_nrg.get('nrg_LL',0):.0f} LH={freq_nrg.get('nrg_LH',0):.0f}"
                f" HL={freq_nrg.get('nrg_HL',0):.0f} HH={freq_nrg.get('nrg_HH',0):.0f})")

        return " | ".join(p)

    def save_history(self, path):
        import json, os as _os
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2, default=str)


# ════════════════════════════════════════════════════════════
#  Prototype-init diagnostic: source classifier weight cosines
# ════════════════════════════════════════════════════════════

def compute_classifier_weight_cosines(classifier, class_names=None):
    """Pairwise cosine of L2-normalized last-Linear weights W [C, D].

    Checks whether initializing target prototypes from W gives enough separation.

    Args:
        classifier: nn.Module (Sequential or a module with a last Linear)
        class_names: list of str, optional

    Returns:
        dict: cos_AD / cos_DN / cos_AN, pcos_max, pcos_mean (off-diagonal)
    """
    if class_names is None:
        class_names = ['AMD', 'DME', 'NORMAL']

    # Find the last Linear by walking all submodules
    W = None
    for m in classifier.modules():
        if isinstance(m, torch.nn.Linear):
            W = m.weight  # the last Linear overwrites previous ones
    if W is None:
        # Fallback: direct weight attribute
        if hasattr(classifier, 'weight'):
            W = classifier.weight
        else:
            raise ValueError("cannot extract the Linear layer weight from classifier")

    # W shape: [num_classes, feat_dim]
    C = W.shape[0]
    W_n = F.normalize(W.detach(), dim=1, eps=1e-8)  # [C, D]
    cos_matrix = (W_n @ W_n.T).cpu()  # [C, C]

    result = {'pcos_max': -1.0, 'pcos_mean': 0.0}
    pair_count = 0
    for i in range(C):
        for j in range(i + 1, C):
            cos_val = float(cos_matrix[i, j])
            pair_name = f'cos_{class_names[i][0]}{class_names[j][0]}'
            result[pair_name] = cos_val
            result['pcos_max'] = max(result['pcos_max'], cos_val)
            result['pcos_mean'] += cos_val
            pair_count += 1
    if pair_count > 0:
        result['pcos_mean'] /= pair_count

    return result


def print_classifier_weight_cosines(classifier, class_names=None):
    """Print the classifier-weight cosine diagnostic."""
    info = compute_classifier_weight_cosines(classifier, class_names)
    pairs = [(k, v) for k, v in info.items() if k.startswith('cos_')]
    pairs_str = '  '.join(f'{k}={v:.4f}' for k, v in sorted(pairs))
    print(f"  [Target] proto-init: src classifier weight cos: {pairs_str}")
    print(f"  [Target] proto-init: pcos_max={info['pcos_max']:.4f}  pcos_mean={info['pcos_mean']:.4f}")
    if info['pcos_max'] < 0.25:
        print(f"  [Target] proto-init: separation GOOD (pcos_max={info['pcos_max']:.3f} < 0.25)")
    elif info['pcos_max'] < 0.35:
        print(f"  [Target] proto-init: separation OK (pcos_max={info['pcos_max']:.3f} < 0.35)")
    else:
        print(f"  [Target] proto-init: separation POOR (pcos_max={info['pcos_max']:.3f} >= 0.35)")
    return info

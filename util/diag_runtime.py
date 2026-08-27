"""DiagV2 runtime: global collector singleton + per-epoch diagnostic hook + history persistence.

Trainers consume the global collector via diag_v2_epoch_hook;
main() calls diag_v2_init to initialize and diag_v2_finalize to save.
"""
import os
import json

import torch
import torch.nn.functional as F

from util.diag_v2 import DiagV2Collector, loss_gradient_conflict

# ── DiagV2 global singleton (initialized in main(), consumed by train_tgt_caco) ──
_diag_v2_collector = None
_diag_v2_grad_conflict = False
_diag_v2_save_path = ''
# ── Repair tracking: last-epoch target-test preds + feats (for AMD->N repaired/new/stubborn) ──
_prev_epoch_preds = None
_prev_epoch_feats = None


def diag_v2_init(args, class_names):
    """Called once in main() to initialize the DiagV2 collector."""
    global _diag_v2_collector, _diag_v2_grad_conflict, _diag_v2_save_path
    if getattr(args, 'use_diag_v2', 0) != 1:
        _diag_v2_collector = None
        return
    _diag_v2_collector = DiagV2Collector(
        num_classes=len(class_names), class_names=class_names)
    _diag_v2_grad_conflict = getattr(args, 'diag_v2_grad_conflict', 0) == 1
    _diag_v2_save_path = getattr(args, 'diag_v2_save', '')
    print("  [DiagV2] enabled (E0-E5 + A1-C2)")
    if _diag_v2_grad_conflict:
        print("  [DiagV2] C1 grad-conflict enabled (cost 1.5-2x)")


def diag_v2_is_enabled():
    return _diag_v2_collector is not None


def diag_v2_collector():
    """Return the current collector (may be None)."""
    return _diag_v2_collector


def diag_v2_epoch_hook(tgt_encoder, classifier, tgt_test_loader, tgt_test_size,
                       num_classes, class_names, epoch_num, save_name,
                       loss_dict=None, shared_params=None):
    """Called every epoch: collect classification + module diagnostics."""
    global _diag_v2_collector, _prev_epoch_preds, _prev_epoch_feats
    if _diag_v2_collector is None:
        return
    tgt_encoder.eval()
    classifier.eval()
    feats_list, logits_list, labels_list, preds_list = [], [], [], []
    with torch.no_grad():
        for inputs, labels in tgt_test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            feat = tgt_encoder(inputs)[0]
            logit, _ = classifier(feat)
            feats_list.append(feat.cpu())
            logits_list.append(logit.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(logit.argmax(dim=1).cpu())
    if not feats_list:
        return
    feats = torch.cat(feats_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    preds = torch.cat(preds_list, dim=0)
    if _diag_v2_grad_conflict and loss_dict is not None and shared_params is not None:
        tgt_encoder.train()
        classifier.train()
        gc = loss_gradient_conflict(loss_dict, shared_params, retain_graph=True)
        if _diag_v2_collector.history:
            _diag_v2_collector.history[-1]['C1_grad_conflict'] = gc
        tgt_encoder.eval()
        classifier.eval()
    epoch_diag = _diag_v2_collector.compute_epoch_diag(
        feats=feats, logits=logits, labels=labels, preds=preds,
        classifier=classifier)
    line = _diag_v2_collector.format_epoch_line(epoch_diag, epoch_num)
    print("  " + line)

    # ── MissAll + Stubborn: audit all AMD->NORMAL errors + track the factory-stubborn batch ──
    try:
        _mis = (labels == 0) & (preds == 2)    # current AMD->N errors (changes every epoch)
        _cor_a = (labels == 0) & (preds == 0)  # currently-correct AMD (control)
        _srcp = {}
        _srcp_path = os.path.join(save_name, 'src_proto.json')
        if os.path.exists(_srcp_path):
            with open(_srcp_path, 'r') as _jf:
                _srcp = json.load(_jf)
        _f_n = F.normalize(feats.view(feats.size(0), -1), dim=1, eps=1e-8)

        def _cos_parts(idx):
            _parts = []
            for _ci, _lab in [('0', 'A'), ('1', 'D'), ('2', 'N')]:
                if _ci in _srcp:
                    _pp = F.normalize(torch.tensor(_srcp[_ci], dtype=_f_n.dtype), dim=0, eps=1e-8)
                    _c = float((_f_n[idx] @ _pp).mean().item()) if idx.any() else float('nan')
                    _parts.append("{}:{:.2f}".format(_lab, _c))
            return "[" + "/".join(_parts) + "]"

        if _mis.any():
            _lgm = logits[_mis].mean(0)
            _cor_str = " (correct-A:{})".format(_cos_parts(_cor_a)) if _cor_a.any() else ""
            print("  [MissAll] AMD->N n={} | logit[A/D/N]=[{:.1f}/{:.1f}/{:.1f}] | src-centroid-cos{}{}".format(
                int(_mis.sum().item()), _lgm[0].item(), _lgm[1].item(), _lgm[2].item(),
                _cos_parts(_mis), _cor_str))
        # ── Factory-stubborn batch tracking (historical anchor) ──
        _stub_path = os.path.join(save_name, 'stubborn_idx.json')
        if os.path.exists(_stub_path):
            with open(_stub_path, 'r') as _jf:
                _stub = [int(x) for x in json.load(_jf)]
            _stub = torch.tensor(_stub, dtype=torch.long)
            _stub = _stub[_stub < len(labels)]
            if len(_stub) > 0:
                _n_still = int((preds[_stub] == 2).sum().item())
                print("  [Stubborn] factory-stubborn AMD n={}: still N={} ({:.0f}%)".format(
                    len(_stub), _n_still, 100.0 * _n_still / len(_stub)))
    except Exception:
        pass

    # ── EnDist: target energy distribution by true class (tests the "AMD lesion -> higher energy" prior) ──
    try:
        _fn = feats.norm(dim=1)
        _fe = -torch.logsumexp(logits, dim=1)  # free energy
        _parts = []
        for _c, _cn in [(0, 'A'), (1, 'D'), (2, 'N')]:
            _mc = labels == _c
            if _mc.any():
                _parts.append("{}:fn={:.1f}/fe={:.1f}".format(
                    _cn, _fn[_mc].mean().item(), _fe[_mc].mean().item()))
        _mis2 = (labels == 0) & (preds == 2)
        _extra = ""
        if _mis2.any():
            _extra = " | mis-AMD->N: fn={:.1f}/fe={:.1f}".format(
                _fn[_mis2].mean().item(), _fe[_mis2].mean().item())
        print("  [EnDist] " + " ".join(_parts) + _extra)
    except Exception:
        pass

    # ── FeatGap: are misclassified AMD really merged with NORMAL in the 2048-D feature space? ──
    # Per-sample similarity: for each misclassified AMD x, max + mean normalized cosine vs all
    # correct AMD and all true NORMAL.
    try:
        _mis_g = (labels == 0) & (preds == 2)
        _cor_g = (labels == 0) & (preds == 0)
        _nor_g = (labels == 2) & (preds == 2)
        if _mis_g.any() and _cor_g.any() and _nor_g.any():
            _f2 = feats.view(feats.size(0), -1)
            _mis_f = F.normalize(_f2[_mis_g], dim=1, eps=1e-8)   # [n_mis, D]
            _cor_f = F.normalize(_f2[_cor_g], dim=1, eps=1e-8)   # [n_cor, D]
            _nor_f = F.normalize(_f2[_nor_g], dim=1, eps=1e-8)   # [n_nor, D]
            _s_cor = _mis_f @ _cor_f.T   # [n_mis, n_cor]
            _s_nor = _mis_f @ _nor_f.T   # [n_mis, n_nor]
            _mx_cor = _s_cor.max(dim=1).values
            _mx_nor = _s_nor.max(dim=1).values
            _mn_cor = _s_cor.mean(dim=1)
            _mn_nor = _s_nor.mean(dim=1)
            _frac_nor_max = float((_mx_nor > _mx_cor).float().mean().item())
            _frac_nor_mean = float((_mn_nor > _mn_cor).float().mean().item())
            _avg_mx_cor = float(_mx_cor.mean().item())
            _avg_mx_nor = float(_mx_nor.mean().item())
            _avg_mn_cor = float(_mn_cor.mean().item())
            _avg_mn_nor = float(_mn_nor.mean().item())
            _cor_fb = _cor_f
            _s_corb = _cor_fb @ _nor_f.T
            _mx_corb = _s_corb.max(dim=1).values
            _mn_corb = _s_corb.mean(dim=1)
            _avg_mx_corb = float(_mx_corb.mean().item())
            _avg_mn_corb = float(_mn_corb.mean().item())
            print("  [FeatGap] mis-AMD vs cor-AMD: max={:.2f} mean={:.2f} | vs true NORMAL: max={:.2f} mean={:.2f} | "
                  "closer-to-NOR(max)={:.0%} (mean)={:.0%} | ref: cor-AMD vs NORMAL max={:.2f} mean={:.2f}".format(
                      _avg_mx_cor, _avg_mn_cor, _avg_mx_nor, _avg_mn_nor,
                      _frac_nor_max, _frac_nor_mean, _avg_mx_corb, _avg_mn_corb))
        else:
            print("  [FeatGap] insufficient: misA={} corA={} nor={}".format(
                int(_mis_g.sum().item()), int(_cor_g.sum().item()), int(_nor_g.sum().item())))
    except Exception:
        pass

    # ── Tracer: track factory-stubborn vs new errors by cosine to the NORMAL centroid ──
    try:
        _stub_t_path = os.path.join(save_name, 'stubborn_idx.json')
        _mis_now = (labels == 0) & (preds == 2)
        _nor_now = (labels == 2) & (preds == 2)
        if os.path.exists(_stub_t_path) and _mis_now.any() and _nor_now.any():
            _stub_t = torch.tensor([int(x) for x in json.load(open(_stub_t_path, 'r'))],
                                   dtype=torch.long)
            _stub_t = _stub_t[_stub_t < len(labels)]
            _f2t = feats.view(feats.size(0), -1)
            _nor_cent = F.normalize(_f2t[_nor_now].mean(dim=0), dim=0, eps=1e-8)
            _mis_idx_now = _mis_now.nonzero().flatten()
            _is_stub = torch.isin(_mis_idx_now, _stub_t)
            if _is_stub.any() or (1 - _is_stub).any():
                def _grp_cos(mask):
                    if mask.numel() == 0:
                        return float('nan'), 0
                    _fi = F.normalize(_f2t[mask], dim=1, eps=1e-8)
                    _c = (_fi @ _nor_cent).mean().item()
                    return _c, int(mask.numel())
                _cs, _ns = _grp_cos(_mis_idx_now[_is_stub])
                _cn, _nn = _grp_cos(_mis_idx_now[~_is_stub])
                print("  [Tracer] mis-group -> NORMAL-centroid cos: factory-stubborn[{}]={:.2f} | new[{}]={:.2f}".format(
                    _ns, _cs, _nn, _cn))
            else:
                print("  [Tracer] no mis-classified or group anomaly")
    except Exception:
        pass

    # ── Rescue v2: layer attribution of AMD->N repairs (source anchor as the fixed reference) ──
    try:
        _is_amd = labels == 0
        _mis_now = _is_amd & (preds == 2)   # this-epoch AMD->N
        if _prev_epoch_preds is not None and len(_prev_epoch_preds) == len(labels):
            _prev_amd = _is_amd & (_prev_epoch_preds == 2)          # last-epoch AMD->N
            _rescued = _prev_amd & (preds != 2)                     # was wrong, now right
            _new_bad = (~_prev_amd) & _mis_now                      # newly wrong
            _still = _prev_amd & _mis_now                           # still wrong (stubborn)
            _rescued_n = int(_rescued.sum().item())
            _new_n = int(_new_bad.sum().item())
            _still_n = int(_still.sum().item())
            _extra = ""
            if _rescued_n > 0 and _prev_epoch_feats is not None and \
                    _prev_epoch_feats.size(0) == feats.size(0):
                _srcp_r = {}
                _sp_r = os.path.join(save_name, 'src_proto.json')
                if os.path.exists(_sp_r):
                    with open(_sp_r, 'r') as _jf:
                        _srcp_r = json.load(_jf)
                if '0' in _srcp_r and '2' in _srcp_r:
                    _muA = F.normalize(torch.tensor(_srcp_r['0'], dtype=feats.dtype), dim=0, eps=1e-8)
                    _muN = F.normalize(torch.tensor(_srcp_r['2'], dtype=feats.dtype), dim=0, eps=1e-8)
                    _f_prev = _prev_epoch_feats.view(_prev_epoch_feats.size(0), -1)[_rescued]
                    _fp_n = F.normalize(_f_prev, dim=1, eps=1e-8)
                    _cA_p = float((_fp_n @ _muA).mean().item())
                    _cN_p = float((_fp_n @ _muN).mean().item())
                    _f_cur = feats.view(feats.size(0), -1)[_rescued]
                    _fc_n = F.normalize(_f_cur, dim=1, eps=1e-8)
                    _cA_c = float((_fc_n @ _muA).mean().item())
                    _cN_c = float((_fc_n @ _muN).mean().item())
                    _lg = logits[_rescued]
                    _a_win = int((_lg[:, 0] > _lg[:, 2]).sum().item())
                    _extra = " | rescued-feat-move: cosA {:.2f}->{:.2f} cosN {:.2f}->{:.2f} | A>N flip:{}/{}".format(
                        _cA_p, _cA_c, _cN_p, _cN_c, _a_win, _rescued_n)
            print("  [Rescue] rescued={:d} new-bad={:d} stubborn={:d}{}".format(
                _rescued_n, _new_n, _still_n, _extra))
        else:
            _mis_cur_n = int(_mis_now.sum().item())
            print("  [Rescue] first epoch: current AMD->N={:d}".format(_mis_cur_n))
    except Exception:
        pass

    _prev_epoch_preds = preds.clone()
    _prev_epoch_feats = feats.clone()

    tgt_encoder.train()
    classifier.train()


def diag_v2_finalize(save_name):
    """Called at the end of training to save the history."""
    global _diag_v2_collector, _diag_v2_save_path
    if _diag_v2_collector is None:
        return
    path = _diag_v2_save_path if _diag_v2_save_path else os.path.join(save_name, 'diag_v2_history.json')
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _diag_v2_collector.save_history(path)
        print("  [DiagV2] history saved: {}".format(path))
    except Exception as e:
        print("  [DiagV2] save failed: {}".format(e))

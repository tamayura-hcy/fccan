"""Target trainers: train_tgt_caco (main path) + train_tgt_baseline_ref (strict reference baseline)."""
import copy
import itertools
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from util.data_utils import ensure_save_dir
from util.energy_uda import EnergyUdaState
from util.eval_utils import test
from util.ang import AngModule
from util.em_loss import entropy_loss, entropy_loss_weighted
from util.caco_loss import caco_catnce_loss
from util.diag_v2 import print_classifier_weight_cosines
from util.diag_runtime import diag_v2_epoch_hook, diag_v2_finalize, diag_v2_collector, diag_v2_is_enabled
from util.lr_schedules import linear_decay_scale, cosine_scale, warmup_cosine_scale, one_cycle_scale


def train_tgt_caco(src_encoder, classifier, tgt_encoder, src_data_loader, tgt_data_loader, save_name,
                   num_epochs=10, tgt_test_loader=None, tgt_test_size=None,
                   num_classes=3, lambda_src=0.0, lambda_caco=0.5, caco_tau=0.07,
                   lambda_tgt_reg=0.0, use_tgt_linear_lr=False, tgt_linear_eta_min_ratio=0.0,
                   tgt_lr_sched='none', tgt_lr_warmup_epochs=3, tgt_lr_peak_ratio=1.0,
                   tgt_optimizer='adamw', tgt_enc_lr=None,
                   class_names=None,
                   lambda_em=1.0, src_class_weights=None,
                   scw_em=0.0, scw_tau=0.0, scw_floor=0.15,
                   scw_ll=0.0, scw_ll_alpha=0.5,
                   use_energy_uda=False, energy_tau=1.0, alpha_ea=0.1, alpha_scon=0.1,
                   scon_mix_lambda=0.5, energy_ema_momentum=0.1,
                   lambda_src_ramp_epochs=0, lambda_src_ramp_start_ratio=0.0,
                   lambda_batch_ang=0.0,
                   clf_lr=None,
                   caco_key_conf=0.0,
                   lambda_llinv=0.0, llinv_alpha=0.5, llinv_tau=3.0, llinv_prob=0.5,
                   ema_teacher=0.0, ema_lambda=0.99, ema_warmup_epochs=0,
                   ema_guide_caco=1.0, ema_guide_warmup=8,
                   use_uema=0, uema_lambda_min=0.9, uema_conf_ref=0.5,
                   save_ema_weights=0, save_which='student',
                   swa=1, swa_start_epoch=8,
                   save_weights=1,
                   ):
    """CaCo target training: contrastive learning + SCW-EM + Energy UDA + Batch ETF angular balance.

    lambda_src: source CE anti-forgetting weight; lambda_caco/caco_tau: CaCo weight and temperature;
    lambda_em + scw_*: entropy minimization and self-consistent weighting; use_energy_uda: SCAL+SCON.
    """
    since = time.time()
    ensure_save_dir(save_name)
    # Clean stale target weights so final eval never loads a previous run's SWA/EMA.
    for _stale in ["swa_encoder.pt", "swa_classifier.pt",
                   "ema_encoder.pt", "ema_classifier.pt"]:
        _stale_p = os.path.join(save_name, _stale)
        if os.path.exists(_stale_p):
            try:
                os.remove(_stale_p)
                print("  [Clean] removed stale {}".format(_stale))
            except OSError as _e:
                print("  [Clean] cannot remove {}: {}".format(_stale, _e))
    tgt_opt = (tgt_optimizer or 'adamw').strip().lower()
    base_lr_tgt = 1e-5
    if tgt_opt == 'adamw':
        lr_tgt = tgt_enc_lr if tgt_enc_lr is not None else base_lr_tgt
        lr_clf = clf_lr if clf_lr is not None else lr_tgt
        wd_tgt = lambda_tgt_reg if lambda_tgt_reg > 0 else 0.0001
        _clf_params = [{'params': classifier.parameters(), 'lr': lr_clf}]
        optimizer_tgt = optim.AdamW(
            _clf_params + [{'params': tgt_encoder.parameters(), 'lr': lr_tgt}],
            lr=lr_tgt, weight_decay=wd_tgt)
    else:
        lr_clf_sgd = 0.0001 if clf_lr is None else clf_lr
        _clf_params = [{'params': classifier.parameters(), 'lr': lr_clf_sgd}]
        optimizer_tgt = optim.SGD(
            _clf_params + [{'params': tgt_encoder.parameters(), 'lr': 0.001}],
            lr=0.001, momentum=0.9)
    base_lrs_tgt = [group['lr'] for group in optimizer_tgt.param_groups]
    src_encoder.eval()
    tgt_encoder.train()
    classifier.train()
    _ramp_ep = int(lambda_src_ramp_epochs)
    _r0 = max(0.0, min(1.0, float(lambda_src_ramp_start_ratio)))
    n_main = int(num_epochs)
    n_tot = n_main
    len_src = len(src_data_loader)
    len_tgt = len(tgt_data_loader)
    epoch_test_accs = []

    # Target config banner
    _opt = tgt_optimizer
    _lr = optimizer_tgt.param_groups[0]['lr']
    _sched = (tgt_lr_sched or 'none').strip().lower()
    _sched_txt = 'constant'
    if _sched == 'linear' or bool(use_tgt_linear_lr):
        _sched_txt = 'linear(eta={})'.format(float(tgt_linear_eta_min_ratio))
    elif _sched == 'cosine':
        _sched_txt = 'cosine(eta={})'.format(float(tgt_linear_eta_min_ratio))
    elif _sched == 'warmup_cosine':
        _sched_txt = 'warmup_cosine(wu={},eta={})'.format(int(tgt_lr_warmup_epochs), float(tgt_linear_eta_min_ratio))
    elif _sched == 'onecycle':
        _sched_txt = 'onecycle(peak={},eta={})'.format(float(tgt_lr_peak_ratio), float(tgt_linear_eta_min_ratio))
    _lam = "src={} caco={} em={}".format(lambda_src, lambda_caco, lambda_em)
    if _ramp_ep > 0:
        _lam += " (src-ramp {}ep {:.2f}x)".format(_ramp_ep, _r0)
    print("  [Target] {} epochs | {} lr={} sched={} | lambdas: {}".format(n_tot, _opt, _lr, _sched_txt, _lam))
    if float(lambda_em) <= 0.0:
        print("  [Target] entropy-minimization OFF")
    _modules = []
    if float(scw_ll) > 0.0:
        _modules.append("shortcut-aware-EM(a={},floor={})".format(float(scw_ll_alpha), float(scw_floor)))
    elif float(scw_em) > 0.0:
        _modules.append("self-consistent-EM(floor={},tau={})".format(float(scw_floor), float(scw_tau)))
    if use_tgt_linear_lr and float(tgt_linear_eta_min_ratio) > 0.0:
        _modules.append("linear-lr(eta_min={})".format(tgt_linear_eta_min_ratio))
    _dev_e = next(tgt_encoder.parameters()).device
    energy_state = None
    if use_energy_uda:
        energy_state = EnergyUdaState(
            num_classes, ema_momentum=float(energy_ema_momentum)).to(_dev_e)
        energy_state.train()
        _modules.append("energy(SCAL+SCON,tau={},scon={})".format(energy_tau, alpha_scon))
    if int(swa) == 1:
        _modules.append("swa(start={})".format(int(swa_start_epoch)))
    if _modules:
        print("  [Target] modules: " + " ".join(_modules))

    # SWA: uniform average of student weights from swa_start_epoch; final eval on averaged weights.
    swa_enc = None
    swa_clf = None
    _swa_n = 0
    if int(swa) == 1:
        from torch.optim.swa_utils import AveragedModel, update_bn  # noqa: F401
        swa_enc = AveragedModel(copy.deepcopy(tgt_encoder))
        swa_clf = AveragedModel(copy.deepcopy(classifier))

    # EMA teacher: EMA copies of tgt_encoder+classifier for smoother pseudo-labels/centroids.
    ema_tgt_encoder = None
    ema_classifier = None
    ema_epoch_accs = []   # per-epoch EMA teacher target acc (for result export)
    _teachers = []
    if float(ema_teacher) > 0.0 or float(ema_guide_caco) > 0.0:
        ema_tgt_encoder = copy.deepcopy(tgt_encoder)
        ema_classifier = copy.deepcopy(classifier)
        for _p in ema_tgt_encoder.parameters():
            _p.requires_grad_(False)
        for _p in ema_classifier.parameters():
            _p.requires_grad_(False)
        _teachers.append("EMA(l={},warmup={},guide_warmup={})".format(
            float(ema_lambda), int(ema_warmup_epochs), int(ema_guide_warmup)))
    if int(use_uema) == 1:
        _teachers.append("uncertainty-EMA({}-{})".format(float(uema_lambda_min), float(ema_lambda)))
    if _teachers:
        print("  [Target] teacher: " + " ".join(_teachers))

    # ANG module (Batch ETF angular balance, used when lambda_batch_ang > 0)
    ang_align = AngModule(
        num_classes=num_classes,
        feat_dim=1024,
    ) if lambda_batch_ang > 0 else None
    if ang_align is not None:
        ang_align = ang_align.to(next(tgt_encoder.parameters()).device)
        print("  [Target] modules: angular-balance(Batch ETF, lambda={})".format(lambda_batch_ang))

    # Prototype-init diagnostic: source classifier weight cosines
    print_classifier_weight_cosines(classifier, class_names)
    # ──────────────────────────
    for epoch in range(n_tot):
        lam_caco_epoch = float(lambda_caco)
        if _ramp_ep > 0:
            _t = min(1.0, float(epoch + 1) / float(_ramp_ep))
            lambda_src_eff = float(lambda_src) * (_r0 + (1.0 - _r0) * _t)
        else:
            lambda_src_eff = float(lambda_src)

        _sched = (tgt_lr_sched or 'none').strip().lower()
        if _sched == 'cosine':
            lr_scale = cosine_scale(epoch, n_tot, float(tgt_linear_eta_min_ratio))
        elif _sched == 'warmup_cosine':
            lr_scale = warmup_cosine_scale(epoch, n_tot, int(tgt_lr_warmup_epochs), float(tgt_linear_eta_min_ratio))
        elif _sched == 'onecycle':
            lr_scale = one_cycle_scale(epoch, n_tot, float(tgt_lr_peak_ratio), float(tgt_linear_eta_min_ratio))
        elif _sched == 'linear' or bool(use_tgt_linear_lr):
            lr_scale = linear_decay_scale(epoch, n_tot, float(tgt_linear_eta_min_ratio))
        else:
            lr_scale = 1.0
        for i, pg in enumerate(optimizer_tgt.param_groups):
            pg['lr'] = base_lrs_tgt[i] * lr_scale
        n_batches = max(len_src, len_tgt)
        if len_src >= len_tgt:
            loader_pairs = zip(src_data_loader, itertools.cycle(tgt_data_loader))
        else:
            loader_pairs = zip(itertools.cycle(src_data_loader), tgt_data_loader)
        pbar = tqdm(loader_pairs, total=n_batches, desc='Epoch [{}/{}]'.format(epoch + 1, n_tot), leave=True)
        for step, ((images_src, label_src), (images_tgt, label_tgt)) in enumerate(pbar):
            if torch.cuda.is_available():
                images_src = images_src.cuda().clone()
                images_tgt = images_tgt.cuda().clone()
                label_src, label_tgt = label_src.cuda(), label_tgt.cuda()
            else:
                images_src = images_src.clone()
                images_tgt = images_tgt.clone()
            optimizer_tgt.zero_grad()
            # Single forward pass
            with torch.no_grad():
                feat_src = src_encoder(images_src)[0]
            feat_tgt, _ = tgt_encoder(images_tgt)

            # Classifier switching
            _clf = classifier
            preds_tgt, _ = _clf(feat_tgt)
            preds_src, _ = _clf(feat_src)
            # EMA teacher forward: after warmup, pseudo-labels/confidence use EMA predictions.
            ema_logits = None
            if ema_tgt_encoder is not None and float(ema_guide_caco) > 0.0 and (epoch + 1) > int(ema_guide_warmup):
                ema_tgt_encoder.eval()
                ema_classifier.eval()
                with torch.no_grad():
                    _ef = ema_tgt_encoder(images_tgt)[0]
                    ema_logits, _ = ema_classifier(_ef)
                ema_tgt_encoder.train()
                ema_classifier.train()
            # Self-consistent weighted entropy (SCW-EM, soft gate): Bhattacharyya coefficient BC of
            # two forward softmaxes as a continuous weight w in [0,1].
            _scg_weight = None
            if float(scw_em) > 0.0 or float(scw_ll) > 0.0:
                with torch.no_grad():
                    if float(scw_ll) > 0.0:
                        # LL shortcut-aware weighting: prediction drift under LL perturbation measures shortcut sensitivity.
                        from util.ll_strength_aug import ll_strength_augment
                        _xa = ll_strength_augment(images_tgt, alpha=float(scw_ll_alpha), p=1.0)
                        _fa, _ = tgt_encoder(_xa)
                        _la, _ = _clf(_fa)
                        _p1 = F.softmax(preds_tgt.detach(), dim=1)
                        _p2 = F.softmax(_la.detach(), dim=1)
                        _drift = 0.5 * (_p1 - _p2).abs().sum(dim=1)
                        _cons = (1.0 - _drift).clamp(min=0.0, max=1.0)
                    else:
                        _logits2, _ = _clf(feat_tgt)
                        _p1 = F.softmax(preds_tgt.detach(), dim=1)
                        _p2 = F.softmax(_logits2, dim=1)
                        # Consistency: Bhattacharyya coefficient
                        _cons = torch.sqrt((_p1 * _p2).clamp_min(1e-12)).sum(dim=1)
                    # Optional confidence lower bound: down-weight low-confidence samples
                    if float(scw_tau) > 0.0:
                        _conf = _p1.max(dim=1).values
                        _cons = _cons * (_conf / float(scw_tau)).clamp(max=1.0)
                    _floor = float(scw_floor)
                    _scg_weight = _floor + (1.0 - _floor) * _cons
            # ---- end SCW-EM weight ----
            if ema_logits is not None:
                pseudo_tgt = ema_logits.argmax(dim=1)
            else:
                pseudo_tgt = preds_tgt.argmax(dim=1)

            # Losses: source CE + entropy + CaCo
            _w = src_class_weights.to(preds_src.device) if src_class_weights is not None else None

            # ---- source cross-entropy ----
            loss_src = F.cross_entropy(preds_src, label_src, weight=_w)

            if _scg_weight is not None:
                loss_em = entropy_loss_weighted(preds_tgt, _scg_weight)
            else:
                loss_em = entropy_loss(preds_tgt)
            # CaCo keys: source + high-confidence target (key-side confidence filter)
            src_k = feat_src.detach()
            # ── Key-side confidence filter ──
            if float(caco_key_conf) > 0.0:
                _pk = ema_logits.softmax(dim=1) if ema_logits is not None else preds_tgt.softmax(dim=1)
                _conf_k = _pk.max(dim=1).values
                _key_mask = _conf_k >= float(caco_key_conf)
                if _key_mask.any():
                    keys = torch.cat([src_k, feat_tgt[_key_mask]], dim=0)
                    key_labels = torch.cat([label_src, pseudo_tgt[_key_mask].detach()], dim=0)
                else:
                    keys = src_k
                    key_labels = label_src
            else:
                keys = torch.cat([src_k, feat_tgt], dim=0)
                key_labels = torch.cat([label_src, pseudo_tgt.detach()], dim=0)
            q = F.normalize(feat_tgt, dim=1, eps=1e-8)
            keys_n = F.normalize(keys, dim=1, eps=1e-8)
            loss_caco = caco_catnce_loss(
                q, keys_n, key_labels, pseudo_tgt, num_classes, tau=caco_tau,
                per_query_weight=None)
            loss = (lambda_src_eff * loss_src + float(lambda_em) * loss_em
                    + lam_caco_epoch * loss_caco)
            # L_llinv: low-frequency invariance (symmetric KL, temperature-softened, self-supervised).
            loss_llinv_val = None
            if float(lambda_llinv) > 0:
                from util.ll_strength_aug import ll_strength_augment
                images_tgt_aug = ll_strength_augment(
                    images_tgt, alpha=float(llinv_alpha), p=float(llinv_prob))
                feat_tgt_aug, _ = tgt_encoder(images_tgt_aug)
                preds_tgt_aug, _ = _clf(feat_tgt_aug)
                _tau = float(llinv_tau)
                _lp1 = F.log_softmax(preds_tgt / _tau, dim=1)
                _sp2 = F.softmax(preds_tgt_aug / _tau, dim=1)
                _kl1 = F.kl_div(_lp1, _sp2, reduction='batchmean')
                _lp2 = F.log_softmax(preds_tgt_aug / _tau, dim=1)
                _sp1 = F.softmax(preds_tgt / _tau, dim=1)
                _kl2 = F.kl_div(_lp2, _sp1, reduction='batchmean')
                loss_llinv = 0.5 * (_kl1 + _kl2)
                loss = loss + float(lambda_llinv) * loss_llinv
                loss_llinv_val = loss_llinv.item()
            loss_align_val = None  # alignment loss removed
            ea_w = None
            loss_ea_val = None
            loss_scon_val = None
            if energy_state is not None:
                loss_ea_b, loss_scon_b, _ = energy_state.forward_losses(
                    preds_src, preds_tgt, tau=float(energy_tau), lambda_scon=float(scon_mix_lambda),
                    ea_weights=ea_w)
                loss = loss + float(alpha_ea) * loss_ea_b + float(alpha_scon) * loss_scon_b
                loss_ea_val = loss_ea_b.item()
                loss_scon_val = loss_scon_b.item()

            # Batch ETF: keep class centroids equiangularly separated
            feat_tgt_n = F.normalize(feat_tgt, dim=1, eps=1e-8)
            loss_batch_ang_val = None
            if lambda_batch_ang > 0:
                batch_means = []
                for c in range(num_classes):
                    mc = (pseudo_tgt == c)
                    if mc.any():
                        batch_means.append(feat_tgt_n[mc].mean(0))
                    else:
                        batch_means.append(torch.zeros(feat_tgt_n.size(1), device=feat_tgt_n.device, dtype=feat_tgt_n.dtype))
                batch_means = torch.stack(batch_means, dim=0)
                loss_batch_ang = ang_align.angular_balance_loss(batch_means, None)
                loss = loss + lambda_batch_ang * loss_batch_ang
                loss_batch_ang_val = loss_batch_ang.item()

            loss.backward()
            optimizer_tgt.step()

            # EMA teacher update
            if ema_tgt_encoder is not None:
                _decay = float(ema_lambda)
                if (epoch + 1) <= int(ema_warmup_epochs):
                    _decay = 1.0  # no update during warmup, keep factory model
                elif int(use_uema) == 1:
                    # UEMA: adapt lambda by target entropy; high entropy lowers lambda so the teacher tracks faster.
                    with torch.no_grad():
                        _p_soft = F.softmax(preds_tgt.detach(), dim=1)
                        _ent = -(_p_soft * (torch.log(_p_soft.clamp_min(1e-12)))).sum(dim=1).mean()
                        _ent_max = float(torch.log(torch.tensor(max(2, num_classes), dtype=_ent.dtype, device=_ent.device)))
                        _unc = (_ent / _ent_max).clamp(0.0, 1.0).item()  # normalized uncertainty [0,1]
                        _conf = 1.0 - _unc
                    # lambda = lambda_min + (lambda_max - lambda_min) * min(1, conf/conf_ref)
                    _lambda_min = float(uema_lambda_min)
                    _conf_ref = max(1e-6, float(uema_conf_ref))
                    _w = min(1.0, _conf / _conf_ref)
                    _decay = _lambda_min + (float(ema_lambda) - _lambda_min) * _w
                with torch.no_grad():
                    for _ep, _pp in zip(ema_tgt_encoder.parameters(), tgt_encoder.parameters()):
                        _ep.mul_(_decay).add_(_pp.detach(), alpha=1.0 - _decay)
                    for _eb, _bb in zip(ema_tgt_encoder.buffers(), tgt_encoder.buffers()):
                        if _eb.dtype.is_floating_point:
                            _eb.mul_(_decay).add_(_bb.detach(), alpha=1.0 - _decay)
                    for _cp, _pc in zip(ema_classifier.parameters(), _clf.parameters()):
                        _cp.mul_(_decay).add_(_pc.detach(), alpha=1.0 - _decay)
                    for _cb, _bc in zip(ema_classifier.buffers(), _clf.buffers()):
                        if _cb.dtype.is_floating_point:
                            _cb.mul_(_decay).add_(_bc.detach(), alpha=1.0 - _decay)

            pf = dict(L_src=round(loss_src.item(), 3), L_em=round(loss_em.item(), 4),
                      L_caco=round(loss_caco.item(), 4))
            if _ramp_ep > 0:
                pf['λsrc'] = round(float(lambda_src_eff), 5)
            if loss_ea_val is not None:
                pf['L_ea'] = round(loss_ea_val, 4)
            if loss_scon_val is not None:
                pf['L_scon'] = round(loss_scon_val, 4)
            if loss_llinv_val is not None:
                pf['L_llinv'] = round(loss_llinv_val, 4)
            if loss_batch_ang_val is not None:
                pf['L_bang'] = round(loss_batch_ang_val, 4)
            if loss_align_val is not None:
                pf['L_align'] = round(loss_align_val, 4)
            pbar.set_postfix(pf)
            del images_src, images_tgt, label_src, label_tgt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if tgt_test_loader is not None and tgt_test_size is not None:
            test_acc, _ = test(
                tgt_encoder, _clf, tgt_test_loader, tgt_test_size,
                show_progress=False, report_per_class=True,
                num_classes=num_classes, class_names=class_names)
            epoch_test_accs.append(float(test_acc))
            print("  [Target test] epoch {}/{} student_acc={:.4f}".format(
                epoch + 1, n_tot, float(test_acc)))
            if ema_tgt_encoder is not None:
                ema_tgt_encoder.eval()
                ema_classifier.eval()
                ema_test_acc, _ = test(
                    ema_tgt_encoder, ema_classifier, tgt_test_loader, tgt_test_size,
                    show_progress=False, report_per_class=True,
                    num_classes=num_classes, class_names=class_names)
                print("  [EMA test]   epoch {}/{} ema_acc={:.4f}".format(
                    epoch + 1, n_tot, float(ema_test_acc)))
                ema_epoch_accs.append(float(ema_test_acc))
                ema_tgt_encoder.train()
                ema_classifier.train()
            # DiagV2 per-epoch diagnostics hook
            if diag_v2_is_enabled():
                if energy_state is not None:
                    diag_v2_collector().set_energy_state(energy_state)
            diag_v2_epoch_hook(
                tgt_encoder, _clf, tgt_test_loader, tgt_test_size,
                num_classes, class_names, epoch + 1, save_name,
                loss_dict=None, shared_params=None)
            tgt_encoder.train()
            _clf.train()
            if energy_state is not None:
                energy_state.train()
            # SWA update: accumulate student weights each epoch from swa_start_epoch
            if swa_enc is not None and swa_clf is not None and (epoch + 1) >= int(swa_start_epoch):
                swa_enc.update_parameters(tgt_encoder)
                swa_clf.update_parameters(classifier)
                _swa_n += 1
    elapsed_time = time.time() - since
    print("Target Training (CaCo) completed in {:.2f}s".format(elapsed_time))
    # Final SWA evaluation and saving (recalc BN stats on target train, then eval averaged weights)
    swa_final_acc = None
    if swa_enc is not None and swa_clf is not None and tgt_test_loader is not None and tgt_test_size is not None:
        try:
            # Recalc BN running stats on target train; move data to model device
            swa_enc.train()
            with torch.no_grad():
                for _img, _lbl in tgt_data_loader:
                    if torch.cuda.is_available():
                        _img = _img.cuda()
                    swa_enc(_img)
            swa_enc.eval()
            swa_clf.eval()
            swa_final_acc, _ = test(
                swa_enc.module, swa_clf.module, tgt_test_loader, tgt_test_size,
                show_progress=False, report_per_class=True,
                num_classes=num_classes, class_names=class_names)
            print("  [SWA test]   final swa_acc={:.4f} (start_ep={}, n_averaged={})".format(
                float(swa_final_acc), int(swa_start_epoch), int(_swa_n)))
        except Exception as _e:
            print("  [SWA test]   skipped: {}".format(_e))
        if int(save_weights) == 1:
            if _swa_n > 0:
                torch.save(swa_enc.module.state_dict(), save_name + '/swa_encoder.pt')
                torch.save(swa_clf.module.state_dict(), save_name + '/swa_classifier.pt')
                print("  [Save] SWA weights saved: swa_encoder.pt / swa_classifier.pt")
            else:
                # No SWA weights averaged (epochs < swa_start_epoch): skip saving to avoid a
                # meaningless "empty SWA" being loaded by main.py final eval.
                print("  [Save] SWA averaged count=0 (epochs < swa_start_epoch={}), skip saving swa_encoder.pt".format(
                    int(swa_start_epoch)))
    # Weight saving: skip all when save_weights=0; else save student/EMA per save_which.
    if int(save_weights) == 1:
        # Default saves student; save_which='ema' saves the EMA teacher instead.
        _save_enc = tgt_encoder
        _save_clf = _clf
        if ema_tgt_encoder is not None and str(save_which).strip().lower() == 'ema':
            _save_enc = ema_tgt_encoder
            _save_clf = ema_classifier
            print("  [Save] saving EMA teacher weights")
        torch.save(_save_enc.state_dict(), save_name + '/target_encoder.pt')
        torch.save(_save_clf.state_dict(), save_name + '/classifier.pt')
        # Always save the EMA copy when the EMA teacher exists; EMA is the default final metric.
        if ema_tgt_encoder is not None:
            torch.save(ema_tgt_encoder.state_dict(), save_name + '/ema_encoder.pt')
            torch.save(ema_classifier.state_dict(), save_name + '/ema_classifier.pt')
            print("  [Save] EMA weights saved: ema_encoder.pt / ema_classifier.pt")
        if energy_state is not None:
            torch.save(energy_state.state_dict(), save_name + '/energy_uda_state.pt')
    # Save DiagV2 history after training
    diag_v2_finalize(save_name)
    return tgt_encoder, classifier, elapsed_time, epoch_test_accs, ema_epoch_accs


# ---- Baseline-only losses for strict reference comparison, not used in the CaCo main path ----
# Original reference adversarial losses, kept only for the strict baseline.

def _baseline_domain_loss(domain_pred, domain_target):
    """Reference baseline: BCE after argmax (domain 0=src 1=tgt)."""
    _, domain_pred = domain_pred.max(1)
    return nn.BCELoss()(domain_pred.to(torch.float), domain_target.to(torch.float))


def _baseline_class_alignment_loss(x_src, x_tgt, pseudo_classes, classes, normalize=True, ca_temperature=None):
    """Reference baseline: class-centroid alignment L_ca (soft weights when ca_temperature>0)."""
    if ca_temperature is None or float(ca_temperature) <= 0.0:
        x_source_copy = x_src.clone()
        x_target_copy = x_tgt.clone()
        pseudo_classes_target_copy = pseudo_classes.clone()
        pseudo_classes_target_copy = torch.argmax(pseudo_classes_target_copy, dim=1)
        classes_source_copy = classes.clone()
        source_dict = dict(zip(x_source_copy, classes_source_copy))
        final_source_dict = {}
        for key in source_dict:
            counter = 1
            s = key
            for inner_key in source_dict:
                if not torch.all(torch.eq(key, inner_key)) and source_dict[key].item() == source_dict[inner_key].item():
                    counter += 1
                    s = s + inner_key
            final_source_dict[source_dict[key].item()] = s / counter
        target_dict = dict(zip(x_target_copy, pseudo_classes_target_copy))
        final_target_dict = {}
        for key in target_dict:
            counter = 1
            s = key
            for inner_key in target_dict:
                if not torch.all(torch.eq(key, inner_key)) and target_dict[key].item() == target_dict[inner_key].item():
                    counter += 1
                    s = s + inner_key
            final_target_dict[target_dict[key].item()] = s / counter
        sum_dists = 0
        num_aligned = 0
        for key in final_source_dict:
            if key in final_target_dict:
                sum_dists = sum_dists + ((final_source_dict[key] - final_target_dict[key]) ** 2).sum(axis=0)
                num_aligned += 1
        if num_aligned > 0 and normalize:
            sum_dists = sum_dists / num_aligned
        return sum_dists
    T = float(ca_temperature)
    num_classes = int(pseudo_classes.shape[1])
    P = F.softmax(pseudo_classes / T, dim=1)
    sum_dists = pseudo_classes.new_tensor(0.0)
    num_aligned = 0
    for k in range(num_classes):
        mask_src = (classes == k)
        if not mask_src.any():
            continue
        proto_s = x_src[mask_src].mean(dim=0)
        w = P[:, k]
        den = w.sum().clamp_min(1e-8)
        proto_t = (w.unsqueeze(1) * x_tgt).sum(dim=0) / den
        sum_dists = sum_dists + ((proto_s - proto_t) ** 2).sum()
        num_aligned += 1
    if num_aligned > 0 and normalize:
        sum_dists = sum_dists / num_aligned
    return sum_dists


def train_tgt_baseline_ref(src_encoder, classifier, tgt_encoder, netD, src_data_loader, tgt_data_loader, save_name, weight,
                           num_epochs=10, tgt_test_loader=None, tgt_test_size=None,
                           num_classes=3, class_names=None, lambda_em=1.0, ca_temperature=None):
    """Strict reference-baseline target training. lambda_em: EM weight, 0=off; ca_temperature: soft CA temperature."""
    since = time.time()
    ensure_save_dir(save_name)
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.SGD([{'params': classifier.parameters(), 'lr': 0.0001},
                               {'params': tgt_encoder.parameters()}], lr=0.001, momentum=0.9)
    optimizer_critic = optim.SGD(netD.parameters(), lr=0.01, momentum=0.9)
    src_encoder.eval()
    tgt_encoder.train()
    classifier.train()
    netD.train()
    epoch_test_accs = []
    if ca_temperature is not None and float(ca_temperature) > 0:
        print("  [Baseline Target] L_ca soft-pseudo-label temp T={}".format(ca_temperature))

    for epoch in range(num_epochs):
        description = "Epoch [{}/{}]".format(epoch + 1, num_epochs)
        pbar = tqdm(zip(src_data_loader, tgt_data_loader), desc=description)
        for (images_src, label_src), (images_tgt, label_tgt) in pbar:
            with torch.no_grad():
                if torch.cuda.is_available():
                    images_src, images_tgt = Variable(images_src.cuda()), Variable(images_tgt.cuda())
                    label_src, label_tgt = Variable(label_src.cuda()), Variable(label_tgt.cuda())
                else:
                    images_src, images_tgt = Variable(images_src), Variable(images_tgt)
                    label_src, label_tgt = Variable(label_src), Variable(label_tgt)

            optimizer_critic.zero_grad()
            feat_src = src_encoder(images_src)[0]
            feat_tgt = tgt_encoder(images_tgt)[0]
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            pred_concat = netD(feat_concat.detach())
            if torch.cuda.is_available():
                domain_src = Variable(torch.zeros(feat_src.size(0)).long().cuda())
                domain_tgt = Variable(torch.ones(feat_tgt.size(0)).long().cuda())
            else:
                domain_src = Variable(torch.zeros(feat_src.size(0)).long())
                domain_tgt = Variable(torch.ones(feat_tgt.size(0)).long())
            domain_concat = torch.cat((domain_src, domain_tgt), 0)
            loss_critic = criterion(pred_concat, domain_concat)
            loss_critic.backward()
            optimizer_critic.step()

            pred_domain = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_domain == domain_concat).float().mean()

            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()
            feat_src, _ = src_encoder(images_src)
            feat_tgt, _ = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            preds_tgt, mid_out_tgt = classifier(feat_tgt)
            _, mid_out_src = classifier(feat_src)
            pred_domain_tgt = netD(feat_tgt)
            pred_domain_concat = netD(feat_concat)

            if torch.cuda.is_available():
                domain_src = Variable(torch.zeros(feat_src.size(0)).long().cuda())
                domain_tgt = Variable(torch.ones(feat_tgt.size(0)).long().cuda())
                pseudo_domain_tgt = Variable(torch.zeros(feat_tgt.size(0)).long().cuda())
            else:
                domain_src = Variable(torch.zeros(feat_src.size(0)).long())
                domain_tgt = Variable(torch.ones(feat_tgt.size(0)).long())
                pseudo_domain_tgt = Variable(torch.zeros(feat_tgt.size(0)).long())
            domain_concat = torch.cat((domain_src, domain_tgt), 0)

            loss_em = entropy_loss(preds_tgt)
            loss_da = _baseline_domain_loss(pred_domain_concat, domain_concat)
            loss_tgt = criterion(pred_domain_tgt, pseudo_domain_tgt)
            loss_ca = _baseline_class_alignment_loss(mid_out_src, mid_out_tgt, preds_tgt, label_src, ca_temperature=ca_temperature)
            loss = loss_tgt + float(lambda_em) * loss_em + weight * (loss_ca + loss_da)
            loss.backward()
            optimizer_tgt.step()
            pbar.set_postfix(D_loss=loss_critic.item(), G_loss=loss_tgt.item(), EM_loss=loss_em.item(),
                             DA_loss=loss_da.item(), CA_loss=loss_ca.item(), netD_acc=acc.item())

            del images_src, images_tgt, label_src, label_tgt, domain_src, domain_tgt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if tgt_test_loader is not None and tgt_test_size is not None:
            test_acc, _ = test(
                tgt_encoder, classifier, tgt_test_loader, tgt_test_size,
                show_progress=False, report_per_class=True,
                num_classes=num_classes, class_names=class_names)
            epoch_test_accs.append(float(test_acc))
            tgt_encoder.train()
            classifier.train()

    elapsed_time = time.time() - since
    print("Target Training completed in {:.2f}s".format(elapsed_time))
    torch.save(netD.state_dict(), save_name + "/netD.pt")
    torch.save(tgt_encoder.state_dict(), save_name + "/target_encoder.pt")
    torch.save(classifier.state_dict(), save_name + "/classifier.pt")
    return tgt_encoder, classifier, elapsed_time, epoch_test_accs

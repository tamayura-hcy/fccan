"""Source trainers: train_src (improved) + train_src_baseline_ref (strict reference baseline)."""
import copy
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

from util.data_utils import ensure_save_dir
from util.lr_schedules import linear_decay_scale, late_linear_scale, apply_lr_scale


def train_src_baseline_ref(encoder, classifier, dataloader_train, dataloader_val, epochs, save_name, class_weights=None, save_weights=1):
    """Strict reference-baseline source training.

    class_weights: per-class CE weights (C,) or None; save_weights=0 skips saving .pt files.
    """
    since = time.time()
    ensure_save_dir(save_name)
    optimizer = optim.SGD([
        {'params': encoder.parameters()},
        {'params': classifier.parameters()}
    ], lr=0.01, momentum=0.9)

    def _ce_ref(logits, lab):
        w = class_weights.to(logits.device) if class_weights is not None else None
        return F.cross_entropy(logits, lab, weight=w)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    train_num = len(dataloader_train.dataset)
    val_num = len(dataloader_val.dataset)
    best_encoder = copy.deepcopy(encoder.state_dict())
    best_classifier = copy.deepcopy(classifier.state_dict())
    best_acc = 0.0

    pbar = tqdm(range(epochs), desc='Training')
    for _ in pbar:
        encoder.train()
        classifier.train()
        loss_train, loss_val, acc_train, acc_val = 0, 0, 0, 0

        for inputs, labels in dataloader_train:
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            features, _ = encoder(inputs)
            outputs = classifier(features)
            _, preds = outputs[0].max(1)
            loss = _ce_ref(outputs[0], labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.data.item()
            acc_train += preds.eq(labels).sum().item()

            del inputs, labels, preds, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = loss_train / train_num
        avg_acc = acc_train / train_num
        scheduler.step()
        encoder.eval()
        classifier.eval()

        for inputs, labels in dataloader_val:
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                features = encoder(inputs)[0]
                outputs = classifier(features)
                _, preds = outputs[0].max(1)
                loss = _ce_ref(outputs[0], labels)
                loss_val += loss.data.item()
                acc_val += preds.eq(labels).sum().item()

            del inputs, labels, preds, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss_val = float(loss_val) / val_num
        avg_acc_val = float(acc_val) / val_num
        pbar.set_postfix(train_acc=avg_acc, train_loss=avg_loss, val_acc=avg_acc_val, val_loss=avg_loss_val)

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
        # Original logic keeps the last epoch for strict-baseline parity.
        best_encoder = copy.deepcopy(encoder.state_dict())
        best_classifier = copy.deepcopy(classifier.state_dict())

    elapsed_time = time.time() - since
    print("Source Training completed in {:.2f}s".format(elapsed_time))
    print("Best val acc on Source = {:.4f}".format(best_acc))
    print()
    encoder.load_state_dict(best_encoder)
    classifier.load_state_dict(best_classifier)
    if int(save_weights) == 1:
        torch.save(encoder.state_dict(), save_name + '/source_encoder.pt')
        torch.save(classifier.state_dict(), save_name + '/classifier.pt')
    return encoder, classifier, elapsed_time


def train_src(encoder, classifier, dataloader_train, dataloader_val, epochs, save_name,
              weight_decay=5e-4,
              src_optimizer='adamw', src_adamw_lr=None,
              src_use_lr_sched=False,
              src_adamw_sched='step', src_adamw_eta_min_ratio=0.05, src_adamw_late_hold_frac=0.55,
              src_sched_step_size=5, src_sched_gamma=0.5,
              base_lr=None,
              num_classes=3, src_ce_temperature=5.0, class_weights=None,
              grad_clip_norm=0.0,
              src_ll_aug=0.0, src_ll_alpha=0.5, src_ll_prob=0.5,
              src_swa=0.0, src_swa_start_epoch=None,
              save_weights=1):
    """Improved source-domain training.

    src_ce_temperature>1: PFAN-style, soften the source softmax to slow overconfidence.
    src_ll_aug: LL energy perturbation to break the "low-frequency strength = class" shortcut.
    save_weights=0 skips saving .pt files.
    """
    since = time.time()
    ensure_save_dir(save_name)
    ############################
    # 1. setup network
    ############################
    opt_params = [{'params': encoder.parameters()}, {'params': classifier.parameters()}]
    # Source base LR (for SGD); AdamW can override via src_adamw_lr, None reuses this value.
    if base_lr is None:
        base_lr = 0.0003
    src_opt = (src_optimizer or 'adamw').strip().lower()
    if src_opt == 'adamw':
        lr = src_adamw_lr if src_adamw_lr is not None else base_lr
        optimizer = optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)
    else:
        lr = base_lr
        optimizer = optim.SGD(opt_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    ce_w_cpu = class_weights
    T_ce = float(src_ce_temperature)
    if T_ce <= 0:
        T_ce = 1.0

    def _ce_src(logits, lab):
        w = ce_w_cpu.to(logits.device) if ce_w_cpu is not None else None
        if T_ce != 1.0:
            return F.cross_entropy(logits / T_ce, lab, weight=w)
        return F.cross_entropy(logits, lab, weight=w)

    base_lrs_src = [pg['lr'] for pg in optimizer.param_groups]
    _adamw_sched = (src_adamw_sched or 'step').strip().lower()
    if _adamw_sched == 'cosine':
        print("  [Warning] src_adamw_sched=cosine deprecated, use linear")
        _adamw_sched = 'linear'
    scheduler = None
    if src_use_lr_sched:
        if src_opt == 'sgd':
            scheduler = StepLR(optimizer, step_size=int(src_sched_step_size), gamma=float(src_sched_gamma))
        elif _adamw_sched == 'step':
            scheduler = StepLR(optimizer, step_size=int(src_sched_step_size), gamma=float(src_sched_gamma))
        elif _adamw_sched not in ('linear', 'late_linear', 'none'):
            scheduler = StepLR(optimizer, step_size=int(src_sched_step_size), gamma=float(src_sched_gamma))
    print("  src optimizer={}, lr={}, src_ce_temperature={}".format(src_optimizer, lr, T_ce))
    if not src_use_lr_sched:
        print("  src lr: fixed (no schedule)")
    elif src_opt == 'adamw':
        print("  src AdamW lr sched = {} (eta_min_ratio={}, late_hold_frac={})".format(
            _adamw_sched, src_adamw_eta_min_ratio, src_adamw_late_hold_frac))
    else:
        print("  src SGD StepLR: step_size={}, gamma={}".format(src_sched_step_size, src_sched_gamma))
    _gclip = float(grad_clip_norm) if grad_clip_norm is not None else 0.0
    if _gclip > 0:
        print("  src grad clip: clip_grad_norm_ = {}".format(_gclip))
    ############################
    # 2. train network
    ############################
    train_num = len(dataloader_train.dataset)
    val_num = len(dataloader_val.dataset)
    best_acc = 0.0  # for printing only; the last epoch is used
    # Source-domain SWA averaging
    _src_swa = int(float(src_swa) if src_swa is not None else 0.0)
    if src_swa_start_epoch is not None and int(src_swa_start_epoch) > 0:
        _swa_start = int(src_swa_start_epoch)
    else:
        _swa_start = max(1, epochs // 2 + 1)
    swa_encoder = AveragedModel(encoder) if _src_swa else None
    swa_classifier = AveragedModel(classifier) if _src_swa else None
    if _src_swa:
        print("  src SWA: averaging weights from epoch {} ({} epochs total)".format(_swa_start, epochs))
    for epoch in range(epochs):
        if src_use_lr_sched and src_opt == 'adamw' and scheduler is None:
            if _adamw_sched == 'linear':
                _sc = linear_decay_scale(epoch, epochs, float(src_adamw_eta_min_ratio))
                apply_lr_scale(optimizer, base_lrs_src, _sc)
            elif _adamw_sched == 'late_linear':
                _sc = late_linear_scale(epoch, epochs, float(src_adamw_late_hold_frac), float(src_adamw_eta_min_ratio))
                apply_lr_scale(optimizer, base_lrs_src, _sc)
        encoder.train()
        classifier.train()
        loss_train, loss_val, acc_train, acc_val = 0, 0, 0, 0

        pbar = tqdm(dataloader_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), leave=True)
        total_seen = 0
        for step, (inputs, labels) in enumerate(pbar):
            bs = labels.size(0)
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
            # zero gradients for optimizer
            optimizer.zero_grad()

            # LL energy perturbation: break the "low-frequency strength = class" shortcut.
            if float(src_ll_aug) > 0:
                from util.ll_strength_aug import ll_strength_augment
                inputs = ll_strength_augment(inputs, alpha=float(src_ll_alpha), p=float(src_ll_prob))

            # compute loss for critic
            features, _ = encoder(inputs)

            outputs = classifier(features)
            logits_s, mid_s = outputs[0], outputs[1]
            _, preds = logits_s.max(1)
            loss = _ce_src(logits_s, labels)
            loss.backward()
            if _gclip > 0:
                _params = [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(_params, max_norm=_gclip)
            optimizer.step()
            loss_train += loss.data.item()
            acc_train += preds.eq(labels).sum().item()
            total_seen += bs

            del inputs, labels, preds, features
            torch.cuda.empty_cache()
            pbar.set_postfix(loss=round(loss.item(), 4), acc=round(acc_train / max(1, total_seen), 4))
        avg_loss = loss_train / max(1, len(dataloader_train))
        avg_acc = acc_train / max(1, total_seen)
        if scheduler is not None:
            scheduler.step()
        encoder.eval()
        classifier.eval()

        # Evaluate val accuracy on original images
        for i, data in enumerate(dataloader_val):
            inputs, labels = data

            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                features = encoder(inputs)[0]
                outputs = classifier(features)

                _, preds = outputs[0].max(1)
                loss = _ce_src(outputs[0], labels)

                loss_val += loss.data.item()
                acc_val += preds.eq(labels).sum().item()

            del inputs, labels, preds, features
            torch.cuda.empty_cache()

        avg_loss_val = float(loss_val) / max(1, len(dataloader_val))
        avg_acc_val = float(acc_val) / val_num

        pbar.set_postfix(train_acc=round(avg_acc, 4), train_loss=round(avg_loss, 4),
                         val_acc=round(avg_acc_val, 4), val_loss=round(avg_loss_val, 4))
        print("  [Epoch {}/{}] train_acc={:.4f} train_loss={:.4f} val_acc={:.4f} val_loss={:.4f}".format(
            epoch + 1, epochs, avg_acc, avg_loss, avg_acc_val, avg_loss_val))

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            if int(save_weights) == 1:
                torch.save(encoder.state_dict(), save_name + '/source_encoder_best.pt')
                torch.save(classifier.state_dict(), save_name + '/classifier_best.pt')
        # Source SWA accumulation
        if _src_swa and (epoch + 1) >= _swa_start:
            swa_encoder.update_parameters(encoder)
            swa_classifier.update_parameters(classifier)

    elapsed_time = time.time() - since
    print("Source Training completed in {:.2f}s".format(elapsed_time))
    print("Best val_acc = {:.4f} (target uses last-epoch model)".format(best_acc))
    print()
    if int(save_weights) == 1:
        torch.save(encoder.state_dict(), save_name + '/source_encoder.pt')
        torch.save(classifier.state_dict(), save_name + '/classifier.pt')
        if _src_swa and swa_encoder is not None:
            torch.save(swa_encoder.module.state_dict(), save_name + '/source_encoder_swa.pt')
            torch.save(swa_classifier.module.state_dict(), save_name + '/classifier_swa.pt')
            print("  [Source SWA] saved {}/source_encoder_swa.pt (avg ep{}..{})".format(
                save_name, _swa_start, epochs))
    # Evaluate SRC-SWA on source val
    if _src_swa and swa_encoder is not None:
        swa_encoder.module.eval()
        swa_classifier.module.eval()
        _swa_correct = 0
        _swa_total = 0
        with torch.no_grad():
            for _inp, _lab in dataloader_val:
                if torch.cuda.is_available():
                    _inp, _lab = _inp.cuda(), _lab.cuda()
                _f = swa_encoder.module(_inp)[0]
                _o = swa_classifier.module(_f)
                _swa_correct += _o[0].max(1)[1].eq(_lab).sum().item()
                _swa_total += _lab.size(0)
        _swa_val_acc = _swa_correct / max(1, _swa_total)
        print("  [Source SWA] val_acc = {:.4f}  (last-epoch val_acc = {:.4f})".format(
            _swa_val_acc, avg_acc_val))
    return encoder, classifier, elapsed_time

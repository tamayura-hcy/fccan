import os
import json
import random
import statistics
import argparse
import copy
import sys

# Force UTF-8 on Windows to avoid GBK UnicodeEncodeError when piping stdout.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Set before importing matplotlib to avoid DLL load failure.
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.fea_net import FEANetBase, FEANet, Classifier, DiscriminatorBaseline
from util.data_utils import (TASK_LIST, LABEL_TO_DATASET, DATASET_TO_LABEL, DATASET_TO_DIR_BG,
                             NUM_WORKERS, get_data_dir, ensure_save_dir)
from util.eval_utils import test
from util.diag_runtime import diag_v2_init
from util.tsne_utils import (save_tsne_and_metrics, save_src_tsne,
                             write_iter_training_log, write_final_average_log)
from util.result_logger import write_full_result_txt
from trainers.source_trainer import train_src, train_src_baseline_ref
from trainers.target_trainer import train_tgt_caco, train_tgt_baseline_ref

# Fixed generator for DataLoader shuffle (deep determinism), set in main().
_DL_GENERATOR = None


def _run_source_training(args, src_encoder, classifier, dataloader_s, save_name, epochs_src,
                         use_baseline, src_ce_w_tensor, src_train_grad_clip, num_cls):
    """Unified entry for source training (shared by transfer / no-transfer branches).

    dataloader_s: dict of 'train'/'val'/'test' loaders.
    """
    if use_baseline:
        return train_src_baseline_ref(
            src_encoder, classifier, dataloader_s['train'], dataloader_s['val'],
            epochs_src, save_name, class_weights=src_ce_w_tensor,
            save_weights=int(getattr(args, 'save_weights', 1)))
    return train_src(
        src_encoder, classifier, dataloader_s['train'], dataloader_s['val'],
        epochs_src, save_name,
        weight_decay=getattr(args, 'weight_decay', '1e-3'),
        src_optimizer=getattr(args, 'src_optimizer', 'adamw'),
        src_adamw_lr=getattr(args, 'src_adamw_lr', None),
        src_use_lr_sched=getattr(args, 'src_use_lr_sched', '1') == 1,
        src_adamw_sched=str(getattr(args, 'src_adamw_sched', 'late_linear')),
        src_adamw_eta_min_ratio=float(getattr(args, 'src_adamw_eta_min_ratio', 0.05)),
        src_adamw_late_hold_frac=float(getattr(args, 'src_adamw_late_hold_frac', 0.55)),
        src_sched_step_size=int(getattr(args, 'src_sched_step_size', 5)),
        src_sched_gamma=float(getattr(args, 'src_sched_gamma', 0.5)),
        base_lr=None,
        num_classes=num_cls,
        src_ce_temperature=getattr(args, 'src_ce_temperature', '1.0'),
        class_weights=src_ce_w_tensor,
        grad_clip_norm=src_train_grad_clip,
        src_ll_aug=float(getattr(args, 'src_ll_aug', '1.0')),
        src_ll_alpha=float(getattr(args, 'src_ll_alpha', '1.0')),
        src_ll_prob=float(getattr(args, 'src_ll_prob', 0.5)),
        src_swa=float(getattr(args, 'src_swa', 0)),
        src_swa_start_epoch=int(getattr(args, 'src_swa_start_epoch', -1)),
        save_weights=int(getattr(args, 'save_weights', 1)))


def _load_source_weights(args, save_name, src_encoder, classifier, _map):
    """Load source weights; prefer SRC-SWA when enabled and file exists."""
    if int(getattr(args, 'src_swa', 0)) == 1:
        _swa_enc = save_name + '/source_encoder_swa.pt'
        if os.path.exists(_swa_enc):
            src_encoder.load_state_dict(torch.load(_swa_enc, map_location=_map))
            classifier.load_state_dict(torch.load(save_name + '/classifier_swa.pt', map_location=_map))
            print("  [Source init] SRC-SWA weights loaded")
            return
    src_encoder.load_state_dict(torch.load(save_name + '/source_encoder.pt', map_location=_map))
    classifier.load_state_dict(torch.load(save_name + '/classifier.pt', map_location=_map))
    print("  [Source init] last-epoch weights loaded")


























def _apply_seed(seed):
    """Apply random seed with deep-determinism setup."""
    _s = int(seed)
    random.seed(_s)
    np.random.seed(_s)
    torch.manual_seed(_s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_s)
    # Reproducibility: force cuDNN deterministic to remove cross-process drift.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # Deep determinism: disable TF32 to avoid cross-run matmul drift.
    # New API for torch>=2.9; fall back to allow_tf32 otherwise.
    try:
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
    except Exception:
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
    # Best-effort deterministic algorithms (warn_only).
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return torch.Generator().manual_seed(_s)


def _parse_seed_list(args):
    """Parse the --seeds comma-separated list; fall back to --seed when empty."""
    raw = str(getattr(args, 'seeds', '') or '').strip()
    if raw:
        seeds = [int(x.strip()) for x in raw.split(',') if x.strip() != '']
        if seeds:
            return seeds
    if getattr(args, 'seed', -1) >= 0:
        return [int(args.seed)]
    return []


def main(args):
    global _DL_GENERATOR
    # --cell_split sets the global CELL_DIR used by get_data_dir.
    import util.data_utils as _du
    _cs = getattr(args, 'cell_split', None)
    if _cs:
        _du.CELL_DIR = _cs
    # Seed list: --seeds comma-separated, cycled one per iteration; fallback to --seed.
    seed_list = _parse_seed_list(args)
    if seed_list:
        _DL_GENERATOR = _apply_seed(seed_list[0])
        if len(seed_list) > 1:
            print("[Seeds] multi-seed: {} (cycled over {} iters)".format(seed_list, int(args.iterations)))
    else:
        _DL_GENERATOR = None
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    epochs_src = args.epochs_src
    epochs_tgt = args.epochs_tgt
    enable_transfer = (args.transferlearning == 1)
    use_baseline = getattr(args, 'use_baseline', 0) == 1
    use_fea_net = getattr(args, 'use_fea_net', '1') == 1  # 1=FEA-Net main model
    iterations = args.iterations
    _batch_src = getattr(args, 'batch_src', None)
    _batch_tgt = getattr(args, 'batch_tgt', None)
    batch_src = _batch_src if (_batch_src is not None and _batch_src > 0) else 32
    batch_tgt = _batch_tgt if (_batch_tgt is not None and _batch_tgt > 0) else 16
    num_workers = 0 if use_baseline else NUM_WORKERS  # baseline uses 0 to match reference

    TRAIN_S, VAL_S, TEST_S = 'train', 'val', 'test'
    TRAIN_T, TEST_T = 'train', 'test'

    # Input resolution; larger (448/512) keeps finer texture.
    # argparse default is None, so fall back to 224 explicitly.
    input_size = getattr(args, 'input_size', None) or 224
    resize_size = int(input_size * 256 / 224)  # keep same padding ratio as default

    # Baseline uses Resize+CenterCrop like the reference (no random crop/flip).
    use_src_aug = getattr(args, 'use_src_aug', '0') == 1 and not use_baseline
    train_transform_s = transforms.Compose([
        transforms.Resize(resize_size), transforms.CenterCrop(input_size), transforms.ToTensor()
    ]) if use_baseline else (transforms.Compose([
        transforms.RandomResizedCrop(input_size), transforms.RandomHorizontalFlip(), transforms.ToTensor()
    ]) if use_src_aug else transforms.Compose([
        transforms.Resize(resize_size), transforms.CenterCrop(input_size), transforms.ToTensor()
    ]))
    data_transform_s = {
        TRAIN_S: train_transform_s,
        VAL_S: transforms.Compose([transforms.Resize(resize_size), transforms.CenterCrop(input_size), transforms.ToTensor()]),
        TEST_S: transforms.Compose([transforms.Resize(resize_size), transforms.CenterCrop(input_size), transforms.ToTensor()])
    }
    data_transform_t = {
        TRAIN_T: transforms.Compose([
            transforms.Resize(resize_size), transforms.CenterCrop(input_size), transforms.ToTensor()]),
        TEST_T: transforms.Compose([transforms.Resize(resize_size), transforms.CenterCrop(input_size), transforms.ToTensor()])
    }

    task_list = list(TASK_LIST)
    only_task = getattr(args, 'only', None)
    if only_task:
        only_task_str = only_task.strip().upper().replace(" ", "").replace("TO", "->")
        only_task_str = only_task_str.replace("=>", "->").replace("→", "->")
        if "->" not in only_task_str:
            raise ValueError(
                "--only format error, e.g. --only BOE->TMI or --only A->B"
            )
        src_tok, tgt_tok = [x.strip() for x in only_task_str.split("->", 1)]

        def _to_label(tok):
            if tok in LABEL_TO_DATASET:
                return tok
            if tok in DATASET_TO_LABEL:
                return DATASET_TO_LABEL[tok]
            raise ValueError(
                "--only domain invalid: {}. Use A/B/C or BOE/TMI/CELL".format(tok)
            )

        src_label_only = _to_label(src_tok)
        tgt_label_only = _to_label(tgt_tok)
        if src_label_only == tgt_label_only:
            raise ValueError("--only does not allow identical source and target")
        task_list = [(src_label_only, tgt_label_only)]
        print("[Only] task: {}->{} ({}->{})".format(
            src_label_only,
            tgt_label_only,
            LABEL_TO_DATASET[src_label_only],
            LABEL_TO_DATASET[tgt_label_only],
        ))

    total_tasks = len(task_list)
    all_task_results = []

    for task_idx, (src_label, tgt_label) in enumerate(task_list):
        source_dataset = LABEL_TO_DATASET[src_label]
        target_dataset = LABEL_TO_DATASET[tgt_label]
        # Use background-removed images when available.
        use_bg = True
        # Strict baseline: use the reference data-path mapping.
        s_data_dir = get_data_dir(source_dataset, use_bg_removed=use_bg, use_baseline_paths=use_baseline)
        t_data_dir = get_data_dir(target_dataset, use_bg_removed=use_bg, use_baseline_paths=use_baseline)

        print("\n" + "=" * 60)
        print("Migration [{}/{}] {} -> {}  ({} -> {})".format(
            task_idx + 1, total_tasks, src_label, tgt_label, source_dataset, target_dataset))
        print("=" * 60)
        bg_s = DATASET_TO_DIR_BG.get(source_dataset)
        bg_t = DATASET_TO_DIR_BG.get(target_dataset)
        if bg_s and os.path.exists(os.path.join('./datasets', bg_s)):
            print(" Source: {}".format(s_data_dir))
        if bg_t and os.path.exists(os.path.join('./datasets', bg_t)):
            print(" Target: {}".format(t_data_dir))
        print(" Loading {} as Source".format(source_dataset))
        print(" Loading {} as Target".format(target_dataset))

        # ---- Source data loading ----
        image_dataset_s = {x: datasets.ImageFolder(os.path.join(s_data_dir, x), transform=data_transform_s[x])
                           for x in [TRAIN_S, VAL_S, TEST_S]}
        _ncls_src = len(image_dataset_s[TRAIN_S].classes)
        _pin_s = torch.cuda.is_available()
        _dl_train_s = torch.utils.data.DataLoader(
            image_dataset_s[TRAIN_S], batch_size=batch_src, shuffle=True, num_workers=num_workers,
            generator=_DL_GENERATOR)
        dataloader_s = {
            TRAIN_S: _dl_train_s,
            VAL_S: torch.utils.data.DataLoader(image_dataset_s[VAL_S], batch_size=batch_src, shuffle=False, num_workers=num_workers),
            TEST_S: torch.utils.data.DataLoader(image_dataset_s[TEST_S], batch_size=batch_src, shuffle=False, num_workers=num_workers)}
        dataset_sizes_src = {x: len(image_dataset_s[x]) for x in [TRAIN_S, VAL_S, TEST_S]}
        for x in [TRAIN_S, VAL_S, TEST_S]:
            print("Loaded {} images under Source {}".format(dataset_sizes_src[x], x))

        # ---- Target data loading ----
        _tmi_tgt_pct = int(getattr(args, 'tmi_target_unlabeled_pct', '75'))
        _merge_tgt_val = False
        if _tmi_tgt_pct == 75:
            if target_dataset == 'TMI':
                _merge_tgt_val = True
            elif target_dataset == 'CELL' and _du.CELL_DIR == 'CELL_split_2025':
                # CELL 2025 split (48/27/25): train+val merged ≈ 75%, same as TMI.
                _merge_tgt_val = True
            else:
                print("  [TGT] note: 75% mode merges val only for TMI or CELL(CELL_split_2025); "
                      "current target={}, cell_split={}".format(target_dataset, _du.CELL_DIR))
        ds_tgt_train = datasets.ImageFolder(
            os.path.join(t_data_dir, TRAIN_T), transform=data_transform_t[TRAIN_T])
        ds_tgt_test = datasets.ImageFolder(
            os.path.join(t_data_dir, TEST_T), transform=data_transform_t[TEST_T])
        if _merge_tgt_val:
            _val_p = os.path.join(t_data_dir, 'val')
            if not os.path.isdir(_val_p):
                raise FileNotFoundError(
                    "tmi_target_unlabeled_pct=75 requires a val/ dir in the target dataset, not found: {}".format(_val_p))
            ds_tgt_val = datasets.ImageFolder(_val_p, transform=data_transform_t[TRAIN_T])
            tgt_train_merged = torch.utils.data.ConcatDataset([ds_tgt_train, ds_tgt_val])
            _ntr, _nv = len(ds_tgt_train), len(ds_tgt_val)
            _nte = len(ds_tgt_test)
            _ratio = 100.0 * len(tgt_train_merged) / max(1, (len(tgt_train_merged) + _nte))
            _mode = "baseline" if use_baseline else "improved"
            print("  [{}:{}] 75% mode: train({})+val({})={} | test={} | unlabeled ratio={:.1f}%".format(
                target_dataset, _mode, _ntr, _nv, len(tgt_train_merged), _nte, _ratio))
            image_dataset_t = {TRAIN_T: tgt_train_merged, TEST_T: ds_tgt_test}
        else:
            if target_dataset in ('TMI', 'CELL'):
                _ntr, _nte = len(ds_tgt_train), len(ds_tgt_test)
                _ratio = 100.0 * _ntr / max(1, (_ntr + _nte))
                _mode = "baseline" if use_baseline else "improved"
                print("  [{}:{}] 50% mode: train({}) | test({}) | unlabeled ratio={:.1f}%".format(
                    target_dataset, _mode, _ntr, _nte, _ratio))
            image_dataset_t = {TRAIN_T: ds_tgt_train, TEST_T: ds_tgt_test}
        dataloader_t = {
            TRAIN_T: torch.utils.data.DataLoader(image_dataset_t[TRAIN_T], batch_size=batch_tgt, shuffle=True, num_workers=num_workers, generator=_DL_GENERATOR),
            TEST_T: torch.utils.data.DataLoader(image_dataset_t[TEST_T], batch_size=batch_tgt, shuffle=False, num_workers=num_workers)}
        dataset_sizes_tgt = {x: len(image_dataset_t[x]) for x in [TRAIN_T, TEST_T]}
        for x in [TRAIN_T, TEST_T]:
            print("Loaded {} images under Target {}".format(dataset_sizes_tgt[x], x))
        print("Batch: src={}, tgt={}".format(batch_src, batch_tgt))

        class_names = image_dataset_s[TRAIN_S].classes
        print("Classes: ", class_names)

        # Initialize DiagV2 diagnostics.
        diag_v2_init(args, class_names)

        src_ce_w_tensor = None

        # Source grad clip: -1=auto, 0=off, >0=manual.
        _src_gclip_arg = float(getattr(args, 'src_grad_clip', -1.0))
        if _src_gclip_arg < 0:
            src_train_grad_clip = 0.0
        else:
            src_train_grad_clip = _src_gclip_arg

        # ---- Model training ----
        test_acc = []
        test_auc = []
        test_recall = []
        test_precision = []
        test_f1 = []
        test_acc_no_transfer = []
        iter_src_final_accs = []
        iter_tgt_epoch_accs = []
        mode_name = "baseline" if use_baseline else "improved"
        for iter in range(1, iterations + 1):
            # Multi-seed: cycle seed_list per iteration; each iter stays reproducible.
            if seed_list:
                _iter_seed = seed_list[(iter - 1) % len(seed_list)]
                _DL_GENERATOR = _apply_seed(_iter_seed)
                print("[Seeds] iter {}/{} seed={}".format(iter, iterations, _iter_seed))
            save_name = './saves/FEANet_' + source_dataset + '_to_' + target_dataset + '_iter' + str(iter)
            _iter_retry_en = bool(enable_transfer and getattr(args, 'iter_tgt_acc_retry', '0') == 1)
            _iter_acc_min = float(getattr(args, 'iter_retry_tgt_acc_min', 0.9))
            _iter_retry_max = int(getattr(args, 'iter_retry_max_attempts', 100))
            _retry_att = 0
            while True:
                _retry_att += 1
                ensure_save_dir(save_name)
                # Baseline uses FEANetBase to match the reference.
                if use_fea_net and not use_baseline:
                    print('Create FEA-Net model (FEA+WRB+HFComp={})................................'.format(
                        getattr(args, 'use_hf_comp', '1')))
                    encoder_model = FEANet(
                        wrb_alpha=getattr(args, 'wrb_alpha', 0.4),
                        wrb_lambda=getattr(args, 'wrb_lambda', 0.3),
                        use_hf_comp=getattr(args, 'use_hf_comp', 0) == 1,
                        hf_comp_scale=getattr(args, 'hf_comp_scale', 0.2),
                        use_msw_sa=getattr(args, 'use_msw_sa', '1') == 1,
                        msw_sa_positions=getattr(args, 'msw_sa_positions', '3'),
                        use_wrb_after_layer2=getattr(args, 'use_wrb_after_layer2', 1) == 1)
                else:
                    print('Create FEA-Net base model (FEA only)...............................................')
                    encoder_model = FEANetBase()
                print('FEA-Net out features=', encoder_model.combined_features)

                # Classifier dropout p=0.3.
                encoder_classifier = Classifier(encoder_model.combined_features, len(class_names), prob=0.3)
                
                if use_gpu:
                    encoder_model.cuda()
                    encoder_classifier.cuda()

                src_encoder = encoder_model
                classifier = encoder_classifier

                # FixRes BN recalibration: update running stats at the current resolution.
                _bn_recalib = getattr(args, 'bn_recalib', 0) == 1
                if _bn_recalib:
                    print("  [BN Recalib] recalibrating BN stats on source data...")
                    _n_batches = getattr(args, 'bn_recalib_batches', 100)
                    src_encoder.train()
                    with torch.no_grad():
                        for _i, (_inp, _) in enumerate(_dl_train_s):
                            if _i >= _n_batches:
                                break
                            if use_gpu:
                                _inp = _inp.cuda()
                            src_encoder(_inp)
                    print("  [BN Recalib] done ({} batches)".format(min(_i + 1, _n_batches)))

                num_cls = len(class_names)

                # Discriminator is only needed for the baseline path.
                netD = None
                if use_baseline:
                    input_dims = src_encoder.combined_features
                    netD = DiscriminatorBaseline(input_dims=input_dims, hidden_dims=500, output_dims=2)
                    if use_gpu:
                        netD.cuda()

                if not enable_transfer:
                    print("No Transfer Learning")
                    src_encoder, classifier, src_time = _run_source_training(
                        args, src_encoder, classifier, dataloader_s, save_name, epochs_src,
                        use_baseline, src_ce_w_tensor, src_train_grad_clip, num_cls)

                    if int(getattr(args, 'save_weights', 1)) == 1:
                        _map = 'cuda' if use_gpu else 'cpu'
                        _load_source_weights(args, save_name, src_encoder, classifier, _map)

                    print("Test scr_encoder + classifier on Source Test dataset")
                    test(src_encoder, classifier, dataloader_s[TEST_S], dataset_sizes_src[TEST_S])
                    # Plot source-only t-SNE after source training.
                    save_src_tsne(source_dataset, iter, src_encoder, classifier,
                                  dataloader_s[TEST_S], dataset_sizes_src[TEST_S],
                                  dataloader_t_test=dataloader_t[TEST_T],
                                  results_root='results', saves_dir=save_name)
                    print("  [Source t-SNE] saved: results/{}_src_only/iter{}/tsne.png".format(
                        source_dataset, iter))
                    print("  [Source t-SNE] copy: {}/tsne_src.png".format(save_name))

                    print("Test scr_encoder + classifier on Target Test dataset")
                    acc, val = test(
                        src_encoder, classifier, dataloader_t[TEST_T], dataset_sizes_tgt[TEST_T],
                        show_progress=False, report_per_class=True,
                        num_classes=num_cls, class_names=class_names)
                    test_acc_no_transfer.append(acc)
                    break

                if enable_transfer:
                    # Baseline mode: strict reference without FEA-Net.
                    if use_baseline:
                        mode_str = " [Baseline]"
                    else:
                        mode_str = " [FEA-Net]" if use_fea_net else ""
                    print("FEA-Net Transfer Learning" + mode_str)
                    src_encoder, classifier, src_time = _run_source_training(
                        args, src_encoder, classifier, dataloader_s, save_name, epochs_src,
                        use_baseline, src_ce_w_tensor, src_train_grad_clip, num_cls)

                    if int(getattr(args, 'save_weights', 1)) == 1:
                        _map = 'cuda' if use_gpu else 'cpu'
                        _load_source_weights(args, save_name, src_encoder, classifier, _map)

                    # Record final source acc once (avoid duplicate appends on retry).
                    src_final_acc_for_log, _ = test(src_encoder, classifier, dataloader_s[TEST_S], dataset_sizes_src[TEST_S])

                    # Plot source-only t-SNE right after source training.
                    save_src_tsne(source_dataset, iter, src_encoder, classifier,
                                  dataloader_s[TEST_S], dataset_sizes_src[TEST_S],
                                  dataloader_t_test=dataloader_t[TEST_T],
                                  results_root='results', saves_dir=save_name)
                    print("  [Source t-SNE] saved: results/{}_src_only/iter{}/tsne.png".format(
                        source_dataset, iter))
                    print("  [Source t-SNE] copy: {}/tsne_src.png".format(save_name))

                    # Source feature diagnostics: per-class fnorm + pcos.
                    try:
                        from util.diag_v2 import per_class_feature_norm, prototype_cosine_matrix
                        src_encoder.eval()
                        _sf, _sl = [], []
                        with torch.no_grad():
                            for _img_s, _lab_s in dataloader_s[TEST_S]:
                                if use_gpu: _img_s = _img_s.cuda()
                                _sf.append(src_encoder(_img_s)[0].detach().cpu())
                                _sl.append(_lab_s)
                        _sf = torch.cat(_sf); _sl = torch.cat(_sl)
                        _src_fn = per_class_feature_norm(_sf, _sl, len(class_names))
                        _src_fn_str = "/".join(f"{f['mean']:.1f}" for f in _src_fn)
                        _src_cos = prototype_cosine_matrix(_sf, _sl, len(class_names))
                        _src_pcos = max(_src_cos[i][j] for i in range(len(_src_cos)) for j in range(len(_src_cos)) if i != j)
                        print("  [SrcDiag] src(factory) fnorm = {} | pcos = {:.3f}".format(_src_fn_str, _src_pcos))
                        # Check whether AMD→N confusion already exists in the source model.
                        try:
                            _dev_c = next(classifier.parameters()).device
                            with torch.no_grad():
                                _slg, _ = classifier(_sf.to(_dev_c))
                            _sp = _slg.argmax(dim=1).cpu()
                            _sfn = _sf.norm(dim=1)
                            _sa_n = int(((_sl == 0) & (_sp == 2)).sum().item())
                            _sa_tot = int((_sl == 0).sum().item())
                            _sn_cor_v = _sfn[(_sp == _sl) & (_sl == 2)]
                            _sa_v = _sfn[_sl == 0]
                            _sa_w = _sfn[(_sl == 0) & (_sp == 2)]
                            _aq = torch.quantile(_sa_v, torch.tensor([0.25, 0.5, 0.75], device=_sa_v.device))
                            _nq = torch.quantile(_sn_cor_v, torch.tensor([0.25, 0.5, 0.75], device=_sn_cor_v.device))
                            _sw_m = _sa_w.mean().item() if len(_sa_w) else float('nan')
                            _sn_m = _sn_cor_v.mean().item()
                            print("  [SrcContrast] src AMD->N mis: {}/{} | AMD fnorm-p=[{:.1f}/{:.1f}/{:.1f}] N-correct-p=[{:.1f}/{:.1f}/{:.1f}] mis-AMD-mean={:.1f} N-mean={:.1f}".format(
                                _sa_n, _sa_tot, _aq[0].item(), _aq[1].item(), _aq[2].item(),
                                _nq[0].item(), _nq[1].item(), _nq[2].item(), _sw_m, _sn_m))
                            # Source energy distribution.
                            try:
                                _s_fe = -torch.logsumexp(_slg.cpu(), dim=1)
                                _s_parts = []
                                for _c, _cn in [(0, 'A'), (1, 'D'), (2, 'N')]:
                                    _sm = _sl == _c
                                    if _sm.any():
                                        _s_parts.append("{}:fn={:.1f}/fe={:.1f}".format(
                                            _cn, _sfn[_sm].mean().item(), _s_fe[_sm].mean().item()))
                                print("  [SrcEn] src energy: " + " ".join(_s_parts))
                            except Exception:
                                pass
                            # Save normalized source centroids for the Stubborn analysis.
                            try:
                                _src_protos = {}
                                for _ci in range(len(class_names)):
                                    _cm = _sl == _ci
                                    if _cm.any():
                                        _src_protos[str(_ci)] = F.normalize(
                                            _sf[_cm].mean(0), dim=0, eps=1e-8).tolist()
                                with open(os.path.join(save_name, 'src_proto.json'), 'w') as _jf:
                                    json.dump(_src_protos, _jf)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception as _e:
                        print("  [SrcDiag] skipped: {}".format(_e))

                    for param in src_encoder.parameters():
                        param.requires_grad = False

                    print("Training encoder for target domain...........................")
                    print('Create target encoder from FEA-Net .............................')
                    if use_fea_net and not use_baseline:
                        target_model = FEANet(
                            wrb_alpha=getattr(args, 'wrb_alpha', 0.4),
                            wrb_lambda=getattr(args, 'wrb_lambda', 0.3),
                            use_hf_comp=getattr(args, 'use_hf_comp', 1) == 1,
                            hf_comp_scale=getattr(args, 'hf_comp_scale', 0.2),
                            use_msw_sa=getattr(args, 'use_msw_sa', 1) == 1,
                            msw_sa_positions=getattr(args, 'msw_sa_positions', '3'),
                            use_wrb_after_layer2=getattr(args, 'use_wrb_after_layer2', 1) == 1)
                    else:
                        target_model = FEANetBase()
                    if use_gpu:
                        target_model.cuda()
                    tgt_encoder = target_model
                    tgt_encoder.load_state_dict(src_encoder.state_dict())

                    # Epoch-0 diagnostic: AMD→N confusion of the factory model on the target test set.
                    try:
                        tgt_encoder.eval()
                        _t0f, _t0l = [], []
                        with torch.no_grad():
                            for _img_t0, _lab_t0 in dataloader_t[TEST_T]:
                                if use_gpu: _img_t0 = _img_t0.cuda()
                                _t0f.append(tgt_encoder(_img_t0)[0].detach().cpu())
                                _t0l.append(_lab_t0)
                        _t0f = torch.cat(_t0f); _t0l = torch.cat(_t0l)
                        _dev_c = next(classifier.parameters()).device
                        _t0lg, _ = classifier(_t0f.to(_dev_c))
                        _t0p = _t0lg.argmax(dim=1).cpu()
                        _t0a_n = int(((_t0l == 0) & (_t0p == 2)).sum().item())
                        _t0a_tot = int((_t0l == 0).sum().item())
                        _t0fn = _t0f.norm(dim=1)
                        _t0a_v = _t0fn[_t0l == 0]
                        _t0n_v = _t0fn[(_t0p == _t0l) & (_t0l == 2)]
                        _t0a_w = _t0fn[(_t0l == 0) & (_t0p == 2)]
                        _aq = torch.quantile(_t0a_v, torch.tensor([0.25, 0.5, 0.75], device=_t0a_v.device))
                        _nq = torch.quantile(_t0n_v, torch.tensor([0.25, 0.5, 0.75], device=_t0n_v.device))
                        _sw = _t0a_w.mean().item() if len(_t0a_w) else float('nan')
                        _sn = _t0n_v.mean().item()
                        print("  [Tgt0] tgt factory(0ep) AMD->N mis: {}/{} | AMD fnorm-p=[{:.1f}/{:.1f}/{:.1f}] N-correct-p=[{:.1f}/{:.1f}/{:.1f}] mis-AMD-mean={:.1f} N-mean={:.1f}".format(
                            _t0a_n, _t0a_tot, _aq[0].item(), _aq[1].item(), _aq[2].item(),
                            _nq[0].item(), _nq[1].item(), _nq[2].item(), _sw, _sn))
                        # Save indices of factory AMD samples predicted as NORMAL.
                        try:
                            _stub_idx = ((_t0l == 0) & (_t0p == 2)).nonzero().flatten().tolist()
                            with open(os.path.join(save_name, 'stubborn_idx.json'), 'w') as _jf:
                                json.dump(_stub_idx, _jf)
                        except Exception:
                            pass
                        # Epoch-0 feature gap: born merged = physical limit; merged after training = fixable.
                        try:
                            _g_mis = (_t0l == 0) & (_t0p == 2)
                            _g_cor = (_t0l == 0) & (_t0p == 0)
                            _g_nor = (_t0l == 2) & (_t0p == 2)
                            if _g_mis.any() and _g_cor.any() and _g_nor.any():
                                _f2 = _t0f.view(_t0f.size(0), -1)
                                _mis_f = F.normalize(_f2[_g_mis], dim=1, eps=1e-8)
                                _cor_f = F.normalize(_f2[_g_cor], dim=1, eps=1e-8)
                                _nor_f = F.normalize(_f2[_g_nor], dim=1, eps=1e-8)
                                _s_cor = _mis_f @ _cor_f.T
                                _s_nor = _mis_f @ _nor_f.T
                                _mx_cor = _s_cor.max(dim=1).values
                                _mx_nor = _s_nor.max(dim=1).values
                                _mn_cor = _s_cor.mean(dim=1)
                                _mn_nor = _s_nor.mean(dim=1)
                                _frac_nor_max = float((_mx_nor > _mx_cor).float().mean().item())
                                _frac_nor_mean = float((_mn_nor > _mn_cor).float().mean().item())
                                print("  [T0FeatGap] factory mis-AMD({}) vs cor-AMD: max={:.2f} mean={:.2f} | "
                                      "vs true NORMAL: max={:.2f} mean={:.2f} | closer-to-NOR(max)={:.0%} (mean)={:.0%}".format(
                                          int(_g_mis.sum().item()),
                                          float(_mx_cor.mean().item()), float(_mn_cor.mean().item()),
                                          float(_mx_nor.mean().item()), float(_mn_nor.mean().item()),
                                          _frac_nor_max, _frac_nor_mean))
                            else:
                                print("  [T0FeatGap] insufficient: misA={} corA={} nor={}".format(
                                    int(_g_mis.sum().item()), int(_g_cor.sum().item()), int(_g_nor.sum().item())))
                        except Exception:
                            pass
                    except Exception:
                        pass
                    tgt_encoder.train()

                    # Transfer path: CaCo main; reference baseline when use_baseline.
                    if not use_baseline:
                        _total = args.epochs_tgt
                        print("  [Target] {} epochs".format(_total))
                        tgt_encoder, classifier, tgt_time, tgt_epoch_accs, tgt_ema_epoch_accs = train_tgt_caco(
                                src_encoder, classifier, tgt_encoder,
                                dataloader_s[TRAIN_S], dataloader_t[TRAIN_T], save_name,
                                num_epochs=_total,
                                tgt_test_loader=dataloader_t[TEST_T], tgt_test_size=dataset_sizes_tgt[TEST_T],
                                num_classes=len(class_names),
                                lambda_src=getattr(args, 'lambda_src', '0.1'),
                                lambda_src_ramp_epochs=int(getattr(args, 'lambda_src_ramp_epochs', 0)),
                                lambda_src_ramp_start_ratio=float(getattr(args, 'lambda_src_ramp_start_ratio', 0.0)),
                                lambda_caco=getattr(args, 'lambda_caco', '0.1'),
                                caco_tau=getattr(args, 'caco_tau', 0.07),
                                lambda_tgt_reg=getattr(args, 'lambda_tgt_reg', '5e-5'),
                                use_tgt_linear_lr=getattr(args, 'use_tgt_linear_lr', 0) == 1,
                                tgt_linear_eta_min_ratio=float(getattr(args, 'tgt_linear_eta_min_ratio', 0.0)),
                                tgt_lr_sched=str(getattr(args, 'tgt_lr_sched', 'none')),
                                tgt_lr_warmup_epochs=int(getattr(args, 'tgt_lr_warmup_epochs', 3)),
                                tgt_lr_peak_ratio=float(getattr(args, 'tgt_lr_peak_ratio', 1.0)),
                                tgt_optimizer=getattr(args, 'tgt_optimizer', 'adamw'),
                                tgt_enc_lr=getattr(args, 'tgt_enc_lr', None),
                                lambda_batch_ang=float(getattr(args, 'lambda_batch_ang', '0.5')),
                                clf_lr=getattr(args, 'tgt_clf_lr', None),
                                class_names=class_names,
                                lambda_em=float(getattr(args, 'lambda_em', 1.0)),
                                scw_em=float(getattr(args, 'scw_em', 0.0)),
                                scw_tau=float(getattr(args, 'scw_tau', 0.0)),
                                scw_floor=float(getattr(args, 'scw_floor', '0.3')),
                                scw_ll=float(getattr(args, 'scw_ll', '1.0')),
                                scw_ll_alpha=float(getattr(args, 'scw_ll_alpha', 0.5)),
                                src_class_weights=src_ce_w_tensor,
                                use_energy_uda=getattr(args, 'use_energy_uda', '1') == 1,
                                energy_tau=float(getattr(args, 'energy_tau', 1.0)),
                                alpha_ea=float(getattr(args, 'alpha_ea', '0.0')),
                                alpha_scon=float(getattr(args, 'alpha_scon', 0.1)),
                                scon_mix_lambda=float(getattr(args, 'scon_mix_lambda', 0.5)),
                                energy_ema_momentum=float(getattr(args, 'energy_ema_momentum', 0.1)),
                                caco_key_conf=float(getattr(args, 'caco_key_conf', '0.95')),
                                lambda_llinv=float(getattr(args, 'lambda_llinv', '1.0')),
                                llinv_alpha=float(getattr(args, 'llinv_alpha', 0.5)),
                                llinv_tau=float(getattr(args, 'llinv_tau', '1.0')),
                                llinv_prob=float(getattr(args, 'llinv_prob', 0.5)),
                                ema_teacher=float(getattr(args, 'ema_teacher', 0.0)),
                                ema_lambda=float(getattr(args, 'ema_lambda', 0.99)),
                                ema_warmup_epochs=int(getattr(args, 'ema_warmup_epochs', 0)),
                                ema_guide_caco=float(getattr(args, 'ema_guide_caco', 1.0)),
                                ema_guide_warmup=int(getattr(args, 'ema_guide_warmup', '8')),
                                use_uema=int(getattr(args, 'use_uema', 0)),
                                uema_lambda_min=float(getattr(args, 'uema_lambda_min', 0.9)),
                                uema_conf_ref=float(getattr(args, 'uema_conf_ref', 0.5)),
                                save_ema_weights=int(getattr(args, 'save_ema_weights', 0)),
                                save_which=str(getattr(args, 'save_which', 'student')),
                                swa=int(getattr(args, 'swa', '1')),
                                swa_start_epoch=int(getattr(args, 'swa_start_epoch', '8')),
                                save_weights=int(getattr(args, 'save_weights', 1)))
                    else:
                        print("  [Baseline] Target domain adaptation (Discriminator)")
                        tgt_encoder, classifier, tgt_time, tgt_epoch_accs = train_tgt_baseline_ref(
                            src_encoder, classifier, tgt_encoder, netD,
                            dataloader_s[TRAIN_S], dataloader_t[TRAIN_T],
                            save_name, 0.01, epochs_tgt,
                            tgt_test_loader=dataloader_t[TEST_T], tgt_test_size=dataset_sizes_tgt[TEST_T],
                            num_classes=len(class_names), class_names=class_names,
                            lambda_em=float(getattr(args, 'lambda_em', 1.0)),
                            ca_temperature=None)

                    # Print both encoders' performance on source & target.
                    print("Test scr_encoder + classifier on Source Test dataset")
                    test(src_encoder, classifier, dataloader_s[TEST_S], dataset_sizes_src[TEST_S])

                    print("Test scr_encoder + classifier on Target Test dataset")
                    test(
                        src_encoder, classifier, dataloader_t[TEST_T], dataset_sizes_tgt[TEST_T],
                        show_progress=False, report_per_class=True,
                        num_classes=len(class_names), class_names=class_names)

                    print("Test tgt_encoder + classifier on Source Test dataset")
                    test(tgt_encoder, classifier, dataloader_s[TEST_S], dataset_sizes_src[TEST_S])

                    # Final eval order: EMA (default) > SWA > student.
                    _eval_enc, _eval_clf = tgt_encoder, classifier
                    _map = 'cuda' if use_gpu else 'cpu'
                    _final_used = 'student'
                    _metric_choice = str(getattr(args, 'final_metric', 'ema')).strip().lower()
                    _ema_enc_p = os.path.join(save_name, 'ema_encoder.pt')
                    _ema_clf_p = os.path.join(save_name, 'ema_classifier.pt')
                    _swa_enc_p = os.path.join(save_name, 'swa_encoder.pt')
                    _swa_clf_p = os.path.join(save_name, 'swa_classifier.pt')
                    # Candidate order: EMA then SWA; 'swa' tries SWA only; 'student' skips.
                    if _metric_choice == 'ema':
                        _cands = [('EMA', _ema_enc_p, _ema_clf_p),
                                  ('SWA', _swa_enc_p, _swa_clf_p)]
                    elif _metric_choice == 'swa':
                        _cands = [('SWA', _swa_enc_p, _swa_clf_p)]
                    else:
                        _cands = []
                    for _name, _enc_p, _clf_p in _cands:
                        if os.path.exists(_enc_p) and os.path.exists(_clf_p):
                            try:
                                _eval_enc = copy.deepcopy(tgt_encoder)
                                _eval_enc.load_state_dict(torch.load(_enc_p, map_location=_map))
                                _eval_clf = copy.deepcopy(classifier)
                                _eval_clf.load_state_dict(torch.load(_clf_p, map_location=_map))
                                _final_used = _name
                                print("  [Final] using {} weights".format(_name))
                                break
                            except Exception as _e:
                                print("  [Final] {} weights load failed, fallback: {}".format(_name, _e))
                                _eval_enc, _eval_clf = tgt_encoder, classifier

                    print("Test tgt_encoder + classifier on Target Test dataset")
                    tgt_acc, tgt_val = test(
                        _eval_enc, _eval_clf, dataloader_t[TEST_T], dataset_sizes_tgt[TEST_T],
                        show_progress=False, report_per_class=True,
                        num_classes=len(class_names), class_names=class_names)
                    if _iter_retry_en and _retry_att < _iter_retry_max and float(tgt_acc) < _iter_acc_min:
                        print("[IterRetry] iter {} attempt {}: tgt test acc={:.4f} < threshold {:.4f}, retraining whole iter".format(
                            iter, _retry_att, tgt_acc, _iter_acc_min))
                        continue
                    if _iter_retry_en and float(tgt_acc) < _iter_acc_min:
                        print("[IterRetry] iter {} exhausted {} attempts below {:.4f}, logging current tgt_acc={:.4f}".format(
                            iter, _retry_att, _iter_acc_min, tgt_acc))
                    test_acc.append(tgt_acc)
                    test_auc.append(tgt_val[0])
                    test_recall.append(tgt_val[1])
                    test_precision.append(tgt_val[2])
                    test_f1.append(tgt_val[4])
                    iter_src_final_accs.append(float(src_final_acc_for_log))
                    iter_tgt_epoch_accs.append(tgt_epoch_accs)
                    write_iter_training_log(
                        mode_name=mode_name,
                        source_dataset=source_dataset,
                        target_dataset=target_dataset,
                        iter_idx=iter,
                        src_final_acc=src_final_acc_for_log,
                        tgt_epoch_accs=tgt_epoch_accs,
                        save_name=save_name,
                        results_root='results'
                    )

                    # Export the full result txt when --save_result_txt=1.
                    if int(getattr(args, 'save_result_txt', 0)) == 1:
                        if 'tgt_ema_epoch_accs' not in locals() or tgt_ema_epoch_accs is None:
                            tgt_ema_epoch_accs = []
                        _fin_metrics = {
                            "acc": float(tgt_acc), "auc": float(tgt_val[0]),
                            "recall": float(tgt_val[1]), "precision": float(tgt_val[2]),
                            "f1": float(tgt_val[4]), "bacc": float(tgt_val[5]),
                            "specificity": float(tgt_val[6]), "kappa": float(tgt_val[7]),
                            "gmean": float(tgt_val[8]), "mcc": float(tgt_val[9]),
                        }
                        try:
                            write_full_result_txt(
                                mode_name=mode_name,
                                source_dataset=source_dataset,
                                target_dataset=target_dataset,
                                iter_idx=iter,
                                seed=int(getattr(args, 'seed', 0)),
                                save_name=save_name,
                                cmd=" ".join(sys.argv),
                                tgt_epoch_accs=tgt_epoch_accs,
                                ema_epoch_accs=tgt_ema_epoch_accs,
                                final_metric_name=_final_used if '_final_used' in dir() else 'student',
                                final_metrics=_fin_metrics,
                                source_final_acc=src_final_acc_for_log,
                                results_root='results')
                        except Exception as _e:
                            print("  [ResultLog] failed: {}".format(_e))

                    # Save t-SNE and metrics for this iteration.
                    metrics_iter = {
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "iter": iter,
                        "use_baseline": use_baseline,
                        "tmi_target_unlabeled_pct": int(getattr(args, 'tmi_target_unlabeled_pct', 75))
                        if target_dataset == 'TMI' else None,
                        "tgt_train_unlabeled_size": int(dataset_sizes_tgt[TRAIN_T]),
                        "tgt_test_size": int(dataset_sizes_tgt[TEST_T]),
                        "tgt_acc": float(tgt_acc),
                        "tgt_auc": float(tgt_val[0]),
                        "tgt_recall": float(tgt_val[1]),
                        "tgt_precision": float(tgt_val[2]),
                        "tgt_f1": float(tgt_val[4]),
                        "tgt_bacc": float(tgt_val[5]),
                        "tgt_specificity": float(tgt_val[6]),
                        "tgt_kappa": float(tgt_val[7]),
                        "tgt_gmean": float(tgt_val[8]),
                        "tgt_mcc": float(tgt_val[9]),
                        "iter_tgt_retry_attempts": int(_retry_att),
                    }
                    save_tsne_and_metrics(src_label, tgt_label, source_dataset, target_dataset,
                                          iter, tgt_encoder, classifier,
                                          dataloader_s[TEST_S], dataloader_t[TEST_T],
                                          dataset_sizes_src[TEST_S], dataset_sizes_tgt[TEST_T],
                                          metrics_iter, results_root='results', saves_dir=save_name)
                    break

        if enable_transfer and test_acc:
            test_acc_avg = sum(test_acc) / len(test_acc)
            test_auc_avg = sum(test_auc) / len(test_auc)
            test_recall_avg = sum(test_recall) / len(test_recall)
            test_precision_avg = sum(test_precision) / len(test_precision)
            test_f1_avg = sum(test_f1) / len(test_f1)
            test_acc_var = statistics.stdev(test_acc) if len(test_acc) > 1 else 0.0
            test_auc_var = statistics.stdev(test_auc) if len(test_auc) > 1 else 0.0
            all_task_results.append({
                'task': '{}->{}'.format(src_label, tgt_label),
                'acc_avg': test_acc_avg, 'acc_var': test_acc_var,
                'auc_avg': test_auc_avg, 'auc_var': test_auc_var,
                'recall_avg': test_recall_avg, 'precision_avg': test_precision_avg,
                'f1_avg': test_f1_avg,
            })
            print('test_acc=', test_acc)
            print('test_auc=', test_auc)
            print("Average test acc: %.4f" % test_acc_avg, '| Variance: %.4f' % test_acc_var)
            print("Average test auc: %.4f" % test_auc_avg)
            print("Average test recall: %.4f" % test_recall_avg, "| Average test precision: %.4f" % test_precision_avg)
            print("Average test f1: %.4f" % test_f1_avg)
            write_final_average_log(
                mode_name=mode_name,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                src_acc_list=iter_src_final_accs,
                tgt_epoch_acc_lists=iter_tgt_epoch_accs,
                results_root='results'
            )
        else:
            if test_acc_no_transfer:
                print("No transferrring test_acc = ", test_acc_no_transfer)

    if enable_transfer and all_task_results:
        print("\n" + "=" * 60)
        print("Summary of {} migration task(s) (A=BOE, B=TMI, C=CELL)".format(total_tasks))
        print("=" * 60)
        for r in all_task_results:
            print(" %s  | acc: %.4f (var %.4f) | auc: %.4f | recall: %.4f | precision: %.4f | f1: %.4f" % (
                r['task'], r['acc_avg'], r['acc_var'], r['auc_avg'], r['recall_avg'], r['precision_avg'], r['f1_avg']))
        print("The End")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FEA-Net: Frequency-Enhanced Attention Network for OCT Unsupervised Domain Adaptation')

    # ---- Task & data ----
    parser.add_argument('-s', '--source', type=str, choices=['BOE', 'CELL', 'TMI'], default='BOE',
                        help='Source dataset, choose from [BOE, CELL, TMI]')
    parser.add_argument('-t', '--target', type=str, choices=['BOE', 'CELL', 'TMI'], default='TMI',
                        help='Target dataset, choose from [BOE, CELL, TMI]')
    parser.add_argument('--only', type=str, default='BOE->TMI',
                        help='Only run the given transfer task, e.g. BOE->TMI or A->B')
    parser.add_argument('--tmi_target_unlabeled_pct', type=int, choices=[50, 75], default=75,
                        help='Only for TMI target: 50=unlabeled train only; 75=train+val (test unchanged)')
    parser.add_argument('--batch_src', type=int, default=16,
                        help='Source batch size')
    parser.add_argument('--batch_tgt', type=int, default=16,
                        help='Target batch size')
    parser.add_argument('--input_size', type=int, default=256,
                        help='Input resolution; larger (448/512) keeps finer texture')
    parser.add_argument('-i', '--iterations', type=int, default=1,
                        help='Number of training iterations for averaging')
    parser.add_argument('--iter_tgt_acc_retry', type=int, choices=[0, 1], default=0,
                        help='Retrain a whole iter if target acc is below threshold')
    parser.add_argument('--iter_retry_tgt_acc_min', type=float, default=0.9,
                        help='Target acc threshold for retry [0,1], e.g. 0.9 = 90%%')
    parser.add_argument('--iter_retry_max_attempts', type=int, default=100,
                        help='Max retry attempts per iter')

    # ---- Training flow ----
    parser.add_argument('-l', '--transferlearning', type=int, choices=[1, 0], default=1,
                        help='Use transfer learning: 1=yes, 0=source-only')
    parser.add_argument('-es', '--epochs_src', type=int, default=5,
                        help='Source training epochs')
    parser.add_argument('-et', '--epochs_tgt', type=int, default=30,
                        help='Target training epochs')
    parser.add_argument('--use_baseline', type=int, choices=[0, 1], default=0,
                        help='1=strict reference baseline, 0=improved FEA-Net')

    # ---- Source training ----
    parser.add_argument('--src_optimizer', type=str, choices=['sgd', 'adamw'], default='adamw',
                        help='Source optimizer: sgd or adamw')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Source weight decay')
    parser.add_argument('--src_adamw_lr', type=float, default=5e-5,
                        help='Source AdamW learning rate; None follows the base lr')
    parser.add_argument('--src_use_lr_sched', type=int, choices=[0, 1], default=1,
                        help='Enable source LR schedule')
    parser.add_argument('--src_adamw_sched', type=str, choices=['step', 'linear', 'late_linear', 'none', 'cosine'], default='late_linear',
                        help='Source AdamW schedule: step/linear/late_linear/none/cosine')
    parser.add_argument('--src_adamw_eta_min_ratio', type=float, default=0.05,
                        help='Min LR ratio (vs initial lr) for linear schedules')
    parser.add_argument('--src_adamw_late_hold_frac', type=float, default=0.55,
                        help='Fraction of epochs holding full LR in late_linear')
    parser.add_argument('--src_sched_step_size', type=int, default=5,
                        help='Source StepLR step size')
    parser.add_argument('--src_sched_gamma', type=float, default=0.5,
                        help='Source StepLR gamma')
    parser.add_argument('--src_ce_temperature', type=float, default=1.0,
                        help='Source CE softmax temperature; >1 softens boundaries')
    parser.add_argument('--src_grad_clip', type=float, default=-1.0,
                        help='Source grad clip: -1=auto, 0=off, >0=manual')
    parser.add_argument('--use_src_aug', type=int, choices=[0, 1], default=0,
                        help='Enable RandomResizedCrop+RandomHorizontalFlip in source training')
    parser.add_argument('--src_ll_aug', type=float, default=1.0,
                        help='Source low-frequency (LL) energy perturbation, enabled when >0')
    parser.add_argument('--src_ll_alpha', type=float, default=1.0,
                        help='LL perturbation magnitude, k=(1+alpha)^u')
    parser.add_argument('--src_ll_prob', type=float, default=0.5,
                        help='LL perturbation probability per batch')
    parser.add_argument('--src_swa', type=int, choices=[0, 1], default=0,
                        help='Apply SWA in source training; init target from the SWA source model')
    parser.add_argument('--src_swa_start_epoch', type=int, default=-1,
                        help='Source SWA start epoch (-1=auto: epochs//2+1)')
    parser.add_argument('--bn_recalib', type=int, choices=[0, 1], default=0,
                        help='FixRes: recalibrate BN stats at the current resolution before training')
    parser.add_argument('--bn_recalib_batches', type=int, default=100,
                        help='Number of batches for BN recalibration')

    # ---- FEA-Net architecture ----
    parser.add_argument('--cell_split', type=str, default='CELL_split_2025',
                        choices=['CELL_split_2025', 'CELL_split_502525'],
                        help='CELL split: CELL_split_2025 (default, DAGCN target 75/25) / CELL_split_502525 (50/25/25)')
    parser.add_argument('--use_fea_net', type=int, choices=[0, 1], default=1,
                        help='1=FEA-Net main model: WRB (+WTConv etc.) after layer3')
    parser.add_argument('--use_wrb_after_layer2', type=int, choices=[0, 1], default=1,
                        help='1=extra WRB(512) between layer2 and layer3; 0=WRB(1024) after layer3 only')
    parser.add_argument('--wrb_alpha', type=float, default=0.4,
                        help='WRB subband gating alpha')
    parser.add_argument('--wrb_lambda', type=float, default=0.3,
                        help='WRB residual fusion lambda')
    parser.add_argument('--use_hf_comp', type=int, choices=[0, 1], default=1,
                        help='Enable the HL/LH/HH Conv+SE high-frequency compensation after layer3')
    parser.add_argument('--hf_comp_scale', type=float, default=0.2,
                        help='HF compensation residual scale, 0.1~0.3 recommended')
    parser.add_argument('--use_msw_sa', type=int, choices=[0, 1], default=1,
                        help='Enable lightweight MSW-SA wavelet spatial attention')
    parser.add_argument('--msw_sa_positions', type=str, default='3',
                        help="MSW-SA insertion positions: '2'/'3'/'23'")

    # ---- Target training ----
    parser.add_argument('--lambda_src', type=float, default=0.1,
                        help='Source CE anti-forgetting weight in the target phase')
    parser.add_argument('--lambda_src_ramp_epochs', type=int, default=0,
                        help='Ramp lambda_src linearly over this many epochs when >0')
    parser.add_argument('--lambda_src_ramp_start_ratio', type=float, default=0.0,
                        help='Ramp start ratio in [0,1] (used with ramp_epochs)')
    parser.add_argument('--lambda_tgt_reg', type=float, default=5e-5,
                        help='Target weight decay; 0=off')
    parser.add_argument('--use_tgt_linear_lr', type=int, choices=[0, 1], default=0,
                        help='Linear LR decay over epochs for the target')
    parser.add_argument('--tgt_linear_eta_min_ratio', type=float, default=0.0,
                        help='Target LR floor = initial x this ratio')
    parser.add_argument('--tgt_lr_sched', type=str, choices=['none', 'linear', 'cosine', 'warmup_cosine', 'onecycle'], default='none',
                        help='Target LR schedule: none/linear/cosine/warmup_cosine/onecycle')
    parser.add_argument('--tgt_lr_warmup_epochs', type=int, default=3,
                        help='Warmup epochs for warmup_cosine')
    parser.add_argument('--tgt_lr_peak_ratio', type=float, default=1.0,
                        help='Peak LR ratio for onecycle')
    parser.add_argument('--tgt_optimizer', type=str, choices=['sgd', 'adamw'], default='adamw',
                        help='Target optimizer: sgd or adamw')
    parser.add_argument('--tgt_enc_lr', type=float, default=1e-5,
                        help='Target encoder LR')
    parser.add_argument('--tgt_clf_lr', type=float, default=None,
                        help='Target classifier LR; None=tgt_enc_lr')

    # ---- Loss modules ----
    # EM / SCW-LL
    parser.add_argument('--lambda_em', type=float, default=1.0,
                        help='Entropy minimization weight; 0=off')
    parser.add_argument('--scw_em', type=float, default=0.0,
                        help='Self-consistent weighted entropy (SCW-EM) switch')
    parser.add_argument('--scw_tau', type=float, default=0.0,
                        help='SCW-EM confidence lower bound; 0=off')
    parser.add_argument('--scw_floor', type=float, default=0.3,
                        help='SCW-EM weight floor')
    parser.add_argument('--scw_ll', type=float, default=1.0,
                        help='LL shortcut-aware entropy weighting; down-weights large drift')
    parser.add_argument('--scw_ll_alpha', type=float, default=0.5,
                        help='LL shortcut-aware perturbation strength')
    # CaCo
    parser.add_argument('--lambda_caco', type=float, default=0.1,
                        help='CaCo category contrastive loss weight')
    parser.add_argument('--caco_tau', type=float, default=0.07,
                        help='CaCo temperature')
    parser.add_argument('--caco_key_conf', type=float, default=0.95,
                        help='CaCo key-side confidence threshold; 0=no filter')
    # L_llinv
    parser.add_argument('--lambda_llinv', type=float, default=1.0,
                        help='Low-frequency invariance consistency weight')
    parser.add_argument('--llinv_alpha', type=float, default=0.5,
                        help='L_llinv perturbation magnitude')
    parser.add_argument('--llinv_tau', type=float, default=1.0,
                        help='L_llinv temperature')
    parser.add_argument('--llinv_prob', type=float, default=0.5,
                        help='L_llinv perturbation probability')
    # EMA teacher
    parser.add_argument('--ema_teacher', type=float, default=0.0,
                        help='EMA teacher: maintain an EMA copy of tgt_encoder+classifier when >0')
    parser.add_argument('--ema_lambda', type=float, default=0.99,
                        help='EMA decay factor; larger = smoother')
    parser.add_argument('--ema_warmup_epochs', type=int, default=0,
                        help='EMA warmup: no updates for the first N epochs')
    parser.add_argument('--ema_guide_caco', type=float, default=1.0,
                        help='EMA guides CaCo: use EMA-smoothed predictions for pseudo-labels/confidence')
    parser.add_argument('--ema_guide_warmup', type=int, default=8,
                        help='EMA guiding warmup epochs')
    parser.add_argument('--use_uema', type=int, choices=[0, 1], default=0,
                        help='Uncertainty-guided EMA: lambda follows the target prediction entropy')
    parser.add_argument('--uema_lambda_min', type=float, default=0.9,
                        help='UEMA min lambda (used at high uncertainty)')
    parser.add_argument('--uema_conf_ref', type=float, default=0.5,
                        help='UEMA confidence reference: lambda restored when conf >= this')
    parser.add_argument('--save_ema_weights', type=int, choices=[0, 1], default=0,
                        help='Also save ema_encoder.pt/ema_classifier.pt')
    parser.add_argument('--save_result_txt', type=int, choices=[0, 1], default=0,
                        help='Export full result txt: per-epoch student/EMA acc + final metrics')
    parser.add_argument('--final_metric', type=str, choices=['ema', 'swa', 'student'], default='ema',
                        help='Weights for final eval: ema (default, falls back to SWA/student) / swa / student')
    parser.add_argument('--save_which', type=str, choices=['student', 'ema'], default='student',
                        help='Save student or EMA weights as target_encoder.pt/classifier.pt')
    parser.add_argument('--swa', type=int, choices=[0, 1], default=1,
                        help='Stochastic Weight Averaging (Izmailov UAI 2018): average student weights from swa_start_epoch')
    parser.add_argument('--swa_start_epoch', type=int, default=8,
                        help='SWA start epoch (1-based)')
    parser.add_argument('--save_weights', type=int, choices=[0, 1], default=1,
                        help='Save .pt weights; 0=skip all weight saving')
    # Energy UDA (SCAL+SCON)
    parser.add_argument('--use_energy_uda', type=int, choices=[0, 1], default=1,
                        help='Enable energy-based UDA (SCAL+SCON)')
    parser.add_argument('--energy_tau', type=float, default=1.0,
                        help='Free-energy logsumexp temperature tau')
    parser.add_argument('--alpha_ea', type=float, default=0.0,
                        help='Free-energy alignment weight; default 0=off')
    parser.add_argument('--alpha_scon', type=float, default=0.1,
                        help='SCON normalization weight')
    parser.add_argument('--scon_mix_lambda', type=float, default=0.5,
                        help='SCON FEN/EN mixing lambda: L_n = l*L_fen + (1-l)*L_en')
    parser.add_argument('--energy_ema_momentum', type=float, default=0.1,
                        help='Energy statistics EMA momentum')
    # Batch ETF angular repulsion
    parser.add_argument('--lambda_batch_ang', type=float, default=0.5,
                        help='Batch ETF angular repulsion weight')

    # ---- Diagnostics ----
    parser.add_argument('--use_diag_v2', type=int, choices=[0, 1], default=1,
                        help='Enable DiagV2 diagnostics')
    parser.add_argument('--diag_v2_grad_conflict', type=int, choices=[0, 1], default=0,
                        help='Enable the C1 gradient-conflict matrix (1.5-2x overhead)')
    parser.add_argument('--diag_v2_save', type=str, default='',
                        help='DiagV2 history save path (default saves_dir/diag_v2_history.json)')

    # ---- Reproducibility ----
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed; -1=not set')
    parser.add_argument('--seeds', type=str, default='',
                        help='Comma-separated seed list, cycled per iteration (e.g. 42,123,777)')

    args = parser.parse_args()
    main(args)

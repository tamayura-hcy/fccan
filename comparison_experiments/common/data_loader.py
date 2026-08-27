"""Unified data loading for comparison experiments, mirroring util/data_utils.py.

All baseline methods share this loader: same data dirs, same ImageFolder format,
same splits, for a fair comparison.
"""
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Windows uses GBK for piped stdout; force UTF-8 to match the reader encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# A=BOE, B=TMI, C=CELL (same as the main project)
TASK_LIST = [
    ('A', 'B'),   # BOE -> TMI
    ('A', 'C'),   # BOE -> CELL
    ('B', 'A'),   # TMI -> BOE
    ('B', 'C'),   # TMI -> CELL
    ('C', 'A'),   # CELL -> BOE
    ('C', 'B'),   # CELL -> TMI
]
LABEL_TO_DATASET = {'A': 'BOE', 'B': 'TMI', 'C': 'CELL'}
DATASET_TO_LABEL = {v: k for k, v in LABEL_TO_DATASET.items()}
# Same as main project DATASET_TO_DIR: CELL uses CELL_split_2025
DATASET_TO_DIR = {'BOE': 'BOE_split_by_person', 'CELL': 'CELL_split_2025', 'TMI': 'TMIdata_split_by_person'}
# CELL split dir (aligned with main project --cell_split); overridable via CELL_SPLIT env var
CELL_DIR = os.environ.get('CELL_SPLIT') or 'CELL_split_2025'
NUM_WORKERS = 0


def get_data_dir(domain, cell_split=None):
    """Return ./datasets/<dir>; CELL uses the given split (default CELL_split_2025)."""
    if domain == 'CELL':
        base = cell_split or CELL_DIR
    else:
        base = DATASET_TO_DIR[domain]
    return os.path.join('./datasets', base)


def build_transforms(input_size=224, train_aug=True, normalize=True):
    """Transform matching thuml/SHOT official: ImageNet normalize + random crop/flip.

    - normalize=True (default): ImageNet mean/std (thuml TLL and SHOT official)
    - normalize=False: DAGCN official has no Normalize (Resize+CenterCrop only)
    - train_aug: train uses RandomHorizontalFlip (thuml official)
    """
    resize = int(input_size * 256 / 224)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train_aug:
        ops = [transforms.Resize(resize), transforms.CenterCrop(input_size),
               transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    else:
        ops = [transforms.Resize(resize), transforms.CenterCrop(input_size),
               transforms.ToTensor()]
    if normalize:
        ops.append(norm)
    return transforms.Compose(ops)


def load_task(src_label, tgt_label, input_size=224, batch_src=16, batch_tgt=16,
              tmi_target_unlabeled_pct=75, cell_split=None, train_aug=True,
              normalize=True):
    """Load source/target train+test loaders for one transfer task, same as main project.

    tmi_target_unlabeled_pct: 75 (default, paper protocol) merges train+val for TMI target;
    50 uses train only.
    cell_split: CELL split dir; default None uses the global CELL_DIR.
    train_aug: RandomHorizontalFlip on source train (off for DAGCN official).
    normalize: ImageNet normalization (off for DAGCN official).

    Returns dict: {src_train, src_test, tgt_train, tgt_test, class_names}
    """
    src_dir = get_data_dir(LABEL_TO_DATASET[src_label], cell_split=cell_split)
    tgt_dir = get_data_dir(LABEL_TO_DATASET[tgt_label], cell_split=cell_split)

    tr_src = build_transforms(input_size, train_aug=train_aug, normalize=normalize)
    tr_tgt = build_transforms(input_size, train_aug=False, normalize=normalize)

    ds_src_train = datasets.ImageFolder(os.path.join(src_dir, 'train'), transform=tr_src)
    ds_src_test = datasets.ImageFolder(os.path.join(src_dir, 'test'), transform=tr_tgt)
    ds_tgt_train = datasets.ImageFolder(os.path.join(tgt_dir, 'train'), transform=tr_tgt)
    ds_tgt_test = datasets.ImageFolder(os.path.join(tgt_dir, 'test'), transform=tr_tgt)
    # source val (for OCT-DDA/DAGCN early stop; None if no val dir)
    ds_src_val = None
    _src_val_p = os.path.join(src_dir, 'val')
    if os.path.isdir(_src_val_p):
        try:
            ds_src_val = datasets.ImageFolder(_src_val_p, transform=tr_tgt)
        except Exception:
            ds_src_val = None

    # Paper protocol (tmi_target_unlabeled_pct=75): target train = train+val (~75%), test = test (25%).
    # TMI target always merges; CELL target merges only for CELL_split_2025 (matches main.py).
    if int(tmi_target_unlabeled_pct) == 75:
        _merge = False
        if tgt_label == 'B':   # TMI
            _merge = True
        elif tgt_label == 'C' and (cell_split or CELL_DIR) == 'CELL_split_2025':
            _merge = True      # CELL + 2025 split: train+val ~75%
        if _merge:
            _val_p = os.path.join(tgt_dir, 'val')
            if os.path.isdir(_val_p):
                ds_tgt_val = datasets.ImageFolder(_val_p, transform=tr_tgt)
                _n_tr, _n_v = len(ds_tgt_train), len(ds_tgt_val)
                ds_tgt_train = torch.utils.data.ConcatDataset([ds_tgt_train, ds_tgt_val])
                print("  [{} tgt] 75% protocol: train{} + val{} = {} (~75%)".format(
                    LABEL_TO_DATASET[tgt_label], _n_tr, _n_v, len(ds_tgt_train)))
            else:
                print("  [{} tgt] 75% protocol needs val/ dir, not found: {} (keep train only)".format(
                    LABEL_TO_DATASET[tgt_label], _val_p))

    dl_src_train = DataLoader(ds_src_train, batch_size=batch_src, shuffle=True,
                              num_workers=NUM_WORKERS)
    dl_src_test = DataLoader(ds_src_test, batch_size=batch_src, shuffle=False,
                             num_workers=NUM_WORKERS)
    dl_tgt_train = DataLoader(ds_tgt_train, batch_size=batch_tgt, shuffle=True,
                              num_workers=NUM_WORKERS)
    dl_tgt_test = DataLoader(ds_tgt_test, batch_size=batch_tgt, shuffle=False,
                             num_workers=NUM_WORKERS)
    dl_src_val = DataLoader(ds_src_val, batch_size=batch_src, shuffle=False,
                            num_workers=NUM_WORKERS) if ds_src_val is not None else None

    return {
        'src_train': dl_src_train, 'src_test': dl_src_test,
        'src_val': dl_src_val,
        'tgt_train': dl_tgt_train, 'tgt_test': dl_tgt_test,
        'class_names': ds_src_train.classes,
    }


def set_seed(seed):
    """Fix random seed, same as main project (incl. deterministic infrastructure)."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass

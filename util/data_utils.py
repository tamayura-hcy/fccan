"""Data paths & generic utilities: task list, dataset-dir mapping, save-dir helper, etc."""
import os
import torch


# A=BOE, B=TMI, C=CELL; 6 migrations: A->B, A->C, B->A, B->C, C->A, C->B
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
# Identical to the DAGCN-Deep-Learning-Project baseline (CELL uses CELL_split_2025)
DATASET_TO_DIR_BASELINE = {'BOE': 'BOE_split_by_person', 'CELL': 'CELL_split_2025', 'TMI': 'TMIdata_split_by_person'}
# CELL uses CELL_split_2025 directly (no OCT2017 fallback)
DATASET_TO_DIR = {'BOE': 'BOE_split_by_person', 'CELL': 'CELL_split_2025', 'TMI': 'TMIdata_split_by_person'}
# CELL split (overridable via main.py --cell_split):
#   'CELL_split_2025' (default, DAGCN target 75/25, train+val) / 'CELL_split_502525' (50/25/25)
CELL_DIR = 'CELL_split_2025'
# Background-removed images: BOE/CELL use bg_removed, TMI uses originals
DATASET_TO_DIR_BG = {'BOE': 'BOE_bg_removed', 'CELL': 'CELL_bg_removed', 'TMI': None}
NUM_WORKERS = 0  # 0 avoids multiprocessing matplotlib load failure; use 4/8 with enough memory


def get_data_dir(domain, use_bg_removed=True, use_baseline_paths=False):
    """Resolve the data directory.

    Baseline mode (use_baseline_paths=True): use DATASET_TO_DIR_BASELINE.
    Improved mode: BOE/CELL prefer the bg-removed images when present; TMI uses originals.
    CELL always uses CELL_DIR (CELL_split_2025 / CELL_split_502525), no OCT2017 fallback.
    """
    if use_baseline_paths:
        base = DATASET_TO_DIR_BASELINE[domain]
        return os.path.join('./datasets', base)
    if domain == 'CELL':
        return os.path.join('./datasets', CELL_DIR)
    base = DATASET_TO_DIR[domain]
    if use_bg_removed and DATASET_TO_DIR_BG.get(domain):
        bg_dir = DATASET_TO_DIR_BG[domain]
        bg_path = os.path.join('./datasets', bg_dir)
        if os.path.exists(bg_path):
            return os.path.join('./datasets', bg_dir)
    return os.path.join('./datasets', base)


def ensure_save_dir(save_name):
    """Create the save directory (incl. ./saves) so torch.save / copy never fail on a missing dir."""
    if not save_name:
        return
    path = os.path.normpath(save_name.strip())
    if not path or path == '.':
        return
    os.makedirs(path, exist_ok=True)


def get_same_class_pair(labels, batch_size):
    """Index of another same-class sample per item (self if none exists); returns idx_p of shape [B]."""
    idx_p = torch.arange(batch_size, device=labels.device)
    for c in labels.unique():
        mask = (labels == c)
        idx_c = torch.where(mask)[0]
        n = idx_c.size(0)
        if n >= 2:
            perm = torch.randperm(n, device=labels.device)
            for i in range(n):
                j = (i + 1) % n
                idx_p[idx_c[perm[i]]] = idx_c[perm[j]]
    return idx_p


def parse_vcreg_layer_weights(spec, expected_n=4):
    """Comma-separated floats, e.g. '1,1,0.5,1'; empty or None means equal weights."""
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    out = [float(x.strip()) for x in s.split(',') if x.strip() != '']
    if len(out) != expected_n:
        raise ValueError(
            "vcreg_layer_weights needs {} floats (comma-separated), got {}: {!r}".format(expected_n, len(out), spec))
    return out

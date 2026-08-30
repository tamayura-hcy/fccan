"""对比实验统一数据加载（仿照主工程 util/data_utils.py）。

（Unified data loading for comparison experiments, mirroring the main project's
util/data_utils.py）

所有对比方法共用本 loader：同样数据目录、同样 ImageFolder 格式、同样划分。
保证对比公平（All baseline methods share this loader for a fair comparison）.
"""
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# 子进程 stdout 是管道时 Windows 默认用 GBK 编码，tee 按 UTF-8 读会乱码；
# 强制 UTF-8，与主脚本 read 编码一致（Force UTF-8 for piped stdout）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# A=BOE, B=TMI, C=CELL；与主工程一致（Same task list as main project）
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
# 与主工程 DATASET_TO_DIR 一致：CELL 直接用 CELL_split_2025
DATASET_TO_DIR = {'BOE': 'BOE_split_by_person', 'CELL': 'CELL_split_2025', 'TMI': 'TMIdata_split_by_person'}
# CELL 划分方式（与主工程 --cell_split 对齐）：
#   'CELL_split_2025'（默认，DAGCN 目标域 75/25）/ 'CELL_split_502525'（50/25/25）
# 可由环境变量 CELL_SPLIT 覆盖（run_comparison.py 透传）
CELL_DIR = os.environ.get('CELL_SPLIT') or 'CELL_split_2025'
NUM_WORKERS = 0


def get_data_dir(domain, cell_split=None):
    """返回 ./datasets/<dir>；CELL 用指定划分（默认 CELL_split_2025），不回退 OCT2017。"""
    if domain == 'CELL':
        base = cell_split or CELL_DIR
    else:
        base = DATASET_TO_DIR[domain]
    return os.path.join('./datasets', base)


def build_transforms(input_size=224, train_aug=True, normalize=True):
    """与 thuml/SHOT 官方一致的 transform：ImageNet Normalize + 随机裁剪/翻转。

    - normalize=True（默认）：ImageNet mean/std [0.485,0.456,0.406]/[0.229,0.224,0.225]
      （thuml Transfer-Learning-Library 与 SHOT 官方均用 ImageNet 归一化）
    - normalize=False：DAGCN 官方无 Normalize（仅 Resize+CenterCrop）
    - train_aug：train 用 RandomHorizontalFlip（thuml 官方 random_horizontal_flip）
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
    """加载一个迁移任务的源/目标 train+test loader，与主工程一致。

    tmi_target_unlabeled_pct：与主工程 main.py 对齐。
      75（默认，论文协议）：TMI 作为目标域时，训练集 = train + val 合并（约 75%），
      与 DAGCN 论文 "training/testing splits = 0.75/0.25" 一致；
      50：仅用 train（约 50%）。
    cell_split：CELL 划分目录（'CELL_split'/'CELL_split_2025'/'CELL_split_502525'），
      默认 None 用全局 CELL_DIR。
    train_aug：源域训练是否用 RandomHorizontalFlip（thuml/SHOT 官方默认开；
      DAGCN 官方无任何增强，其复现需 train_aug=False）。
    normalize：是否 ImageNet 归一化（thuml/SHOT/ADDA 官方默认开；DAGCN 官方无）。

    返回 dict: {src_train, src_test, tgt_train, tgt_test, class_names}
    """
    src_dir = get_data_dir(LABEL_TO_DATASET[src_label], cell_split=cell_split)
    tgt_dir = get_data_dir(LABEL_TO_DATASET[tgt_label], cell_split=cell_split)

    tr_src = build_transforms(input_size, train_aug=train_aug, normalize=normalize)
    tr_tgt = build_transforms(input_size, train_aug=False, normalize=normalize)

    ds_src_train = datasets.ImageFolder(os.path.join(src_dir, 'train'), transform=tr_src)
    ds_src_test = datasets.ImageFolder(os.path.join(src_dir, 'test'), transform=tr_tgt)
    ds_tgt_train = datasets.ImageFolder(os.path.join(tgt_dir, 'train'), transform=tr_tgt)
    ds_tgt_test = datasets.ImageFolder(os.path.join(tgt_dir, 'test'), transform=tr_tgt)
    # 源域 val（OCT-DDA/DAGCN 官方源域 val 早停用；BOE/TMI 有 val 目录，无则 None）
    ds_src_val = None
    _src_val_p = os.path.join(src_dir, 'val')
    if os.path.isdir(_src_val_p):
        try:
            ds_src_val = datasets.ImageFolder(_src_val_p, transform=tr_tgt)
        except Exception:
            ds_src_val = None

    # 论文协议（与主工程 tmi_target_unlabeled_pct=75 一致）：
    #   目标域训练集 = train + val 合并 ≈ 75%，测试 = test（25%）。
    #   TMI（B）目标域：总是合并（DAGCN 论文 0.75/0.25 协议）；
    #   CELL（C）目标域：仅 CELL_split_2025 划分（48/27/25）合并，
    #   与 main.py 的 _merge_tgt_val 逻辑保持一致。
    if int(tmi_target_unlabeled_pct) == 75:
        _merge = False
        if tgt_label == 'B':   # TMI
            _merge = True
        elif tgt_label == 'C' and (cell_split or CELL_DIR) == 'CELL_split_2025':
            _merge = True      # CELL + 2025 划分：train+val≈75%
        if _merge:
            _val_p = os.path.join(tgt_dir, 'val')
            if os.path.isdir(_val_p):
                ds_tgt_val = datasets.ImageFolder(_val_p, transform=tr_tgt)
                _n_tr, _n_v = len(ds_tgt_train), len(ds_tgt_val)
                ds_tgt_train = torch.utils.data.ConcatDataset([ds_tgt_train, ds_tgt_val])
                print("  [{} tgt] 75% 协议: train{} + val{} = {} (约 75%)".format(
                    LABEL_TO_DATASET[tgt_label], _n_tr, _n_v, len(ds_tgt_train)))
            else:
                print("  [{} tgt] 75% 协议需要 val/ 目录，未找到: {}（保持仅 train）".format(
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
    """固定种子，与主工程一致（含深度确定性基建）。"""
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

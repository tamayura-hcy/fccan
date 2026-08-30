# -*- coding: utf-8 -*-
"""make_lists.py —— 为 DaC 官方代码生成三任务数据列表（oct 数据集分支）。

格式（DaC data_list.make_dataset）：每行 "绝对路径 类别号"。
- 源域列表（含标签，供 source.py 预训练）
- 目标域列表（train+val；source-free 适应与评估共用，target.py 会从中划分）

输出：third_party/DaC/VisDA/data/oct/{BOE,TMI,CELL}_list.txt
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VISDA_DIR = os.path.join(ROOT, "comparison_experiments", "third_party", "DaC", "VisDA")
OUT_DIR = os.path.join(VISDA_DIR, "data", "oct")
DATASETS = {
    "BOE": os.path.join(ROOT, "datasets", "BOE_split_by_person"),
    "TMI": os.path.join(ROOT, "datasets", "TMIdata_split_by_person"),
    "CELL": os.path.join(ROOT, "datasets", "CELL_split_2025"),
}
CLASSES = ["AMD", "DME", "NORMAL"]


def collect(ds, splits):
    out = []
    for lab, cls in enumerate(CLASSES):
        for split in splits:
            cls_dir = os.path.join(DATASETS[ds], split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in sorted(os.listdir(cls_dir)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    out.append((os.path.abspath(os.path.join(cls_dir, fn)), lab))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for ds in DATASETS:
        # 源域用 train；目标域用 train+val（DaC 的 target.py 内部划分评估）
        entries = collect(ds, ["train", "val"])
        out_path = os.path.join(OUT_DIR, "{}_list.txt".format(ds))
        with open(out_path, "w", encoding="utf-8") as f:
            for p, lab in entries:
                f.write("{} {}\n".format(p, lab))
        print("{}: n={} -> {}".format(ds, len(entries), out_path))


if __name__ == "__main__":
    main()

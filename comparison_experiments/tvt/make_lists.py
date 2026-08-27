# -*- coding: utf-8 -*-
"""Generate the three-task data lists for the official TVT code.

Format (TVT ImageList/ImageListIndex): one "abs_path label" per line.
Source: train set (labeled); target (UDA unlabeled train): train+val;
test: target test set (labeled, for evaluation).

Task mapping: A->B BOE->TMI; A->C BOE->CELL; B->C TMI->CELL.
Output: lists/{task}/{source_list,target_list,test_list}.txt
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # project root
DATASETS = {
    "BOE": os.path.join(ROOT, "datasets", "BOE_split_by_person"),
    "TMI": os.path.join(ROOT, "datasets", "TMIdata_split_by_person"),
    "CELL": os.path.join(ROOT, "datasets", "CELL_split_2025"),
}
TASKS = [("A-B", "BOE", "TMI"), ("A-C", "BOE", "CELL"), ("B-C", "TMI", "CELL")]
CLASSES = ["AMD", "DME", "NORMAL"]   # alphabetical order -> 0/1/2, same across datasets


def write_list(paths_labels, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for p, lab in paths_labels:
            f.write("{} {}\n".format(p, lab))


def collect(ds, split):
    """Return [(abs_path, label)]; split is 'train' / 'val' / 'test'."""
    out = []
    base = os.path.join(DATASETS[ds], split)
    for lab, cls in enumerate(CLASSES):
        cls_dir = os.path.join(base, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                out.append((os.path.abspath(os.path.join(cls_dir, fn)), lab))
    return out


def main():
    for task, src, tgt in TASKS:
        out_dir = os.path.join(HERE, "lists", task)
        os.makedirs(out_dir, exist_ok=True)
        src_list = collect(src, "train")
        tgt_train = collect(tgt, "train") + collect(tgt, "val")
        tgt_test = collect(tgt, "test")
        write_list(src_list, os.path.join(out_dir, "source_list.txt"))
        write_list(tgt_train, os.path.join(out_dir, "target_list.txt"))
        write_list(tgt_test, os.path.join(out_dir, "test_list.txt"))
        print("{}: source={}  target_trainval={}  test={}".format(
            task, len(src_list), len(tgt_train), len(tgt_test)))


if __name__ == "__main__":
    main()

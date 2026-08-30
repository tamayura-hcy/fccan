# -*- coding: utf-8 -*-
"""make_lists.py —— 为 TVT 官方代码生成三任务数据列表。

格式（TVT ImageList/ImageListIndex）：每行 "绝对路径 类别号"。
- 源域：训练集（含标签）
- 目标域（UDA 无标签训练）：训练+验证集（TVT 的 ImageListIndex 忽略标签，用于聚类）
- 测试：目标域测试集（含标签，用于评估）

任务映射（与主表一致）：
  A->B: BOE -> TMI
  A->C: BOE -> CELL
  B->C: TMI -> CELL

输出：lists/{task}/source_list.txt, target_list.txt, test_list.txt
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # hcy 根目录
DATASETS = {
    "BOE": os.path.join(ROOT, "datasets", "BOE_split_by_person"),
    "TMI": os.path.join(ROOT, "datasets", "TMIdata_split_by_person"),
    "CELL": os.path.join(ROOT, "datasets", "CELL_split_2025"),
}
TASKS = [("A-B", "BOE", "TMI"), ("A-C", "BOE", "CELL"), ("B-C", "TMI", "CELL")]
CLASSES = ["AMD", "DME", "NORMAL"]   # 字母序 → 0/1/2，三数据集一致


def write_list(paths_labels, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for p, lab in paths_labels:
            f.write("{} {}\n".format(p, lab))


def collect(ds, split):
    """返回 [(abs_path, label)]，split 为 'train' / 'val' / 'test'。"""
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

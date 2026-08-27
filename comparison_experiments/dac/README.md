# DaC (NeurIPS 2022) 复现说明

官方代码已 clone 至 `comparison_experiments/third_party/DaC`（https://github.com/ZyeZhang/DaC）。
已为 `source.py` / `target.py` 添加 `--dset oct` 分支（BOE/TMI/CELL，3 类），不影响原有分支。

## 运行（服务器）

```bash
cd comparison_experiments/dac
python make_lists.py                 # 生成 third_party/DaC/VisDA/data/oct/{BOE,TMI,CELL}_list.txt
python run_all.py                    # 三任务 × 5 种子，两阶段（source 预训练 + source-free 适应）
```

## 协议说明

- DaC 是 **source-free** 方法：阶段 1 用源域训练集预训练 ResNet-50；阶段 2 只用目标域数据适应
- 目标域使用训练+验证集（target.py 内部按官方协议划分评估）
- 超参数沿用官方 `run_source.sh` / `run_target.sh`（resnet50、cls_par 0.6、p_threshold 0.97、EMMD）
- 结果解析：日志中的 `Accuracy of the network on the ... test images: X%`

## 已知事项

- DaC 官方为 NeurIPS 2022（非 2023-2026），是否写入论文对比表待定，先跑数据
- target.py 评估时可能随机抽取目标子集，逐种子结果按官方协议报告

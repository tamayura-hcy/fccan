# TVT (WACV 2023) 复现说明

官方代码已 clone 至 `comparison_experiments/third_party/TVT`（https://github.com/uta-smile/TVT）。

## 前置准备（服务器）

1. **预训练权重**：下载 ViT-B_16 (ImageNet-21K) 到 `third_party/TVT/checkpoint/ViT-B_16.npz`
   - 链接见官方 README（Google Cloud Storage `vit_models/imagenet21k/ViT-B_16.npz`，约 340MB）
2. **依赖**：官方要求 torch 1.8.1 + **apex**。服务器为 torch 2.8：
   - 方案 A：`conda install -c conda-forge nvidia-apex`（若编译失败用方案 B）
   - 方案 B：把 `main.py` 顶部的 apex import 替换为 torch 原生 amp/DP（需小幅改代码，可再找我出补丁脚本）

## 运行

```bash
cd comparison_experiments/tvt
python make_lists.py                 # 生成三任务列表（lists/A-B/ 等）
python run_all.py                    # 三任务 × 5 种子，输出 results_tvt.csv
# 单任务调试：
python train.py --task A-B --seed 42
```

## 协议说明

- 数据集用我们统一的 patient-level 划分（BOE/TMI/CELL ImageFolder），列表由 make_lists.py 生成
- 源域用训练集（含标签）；目标域用训练+验证集（无标签使用）；评估在目标域测试集
- 输入 224×224、num_steps=5000（官方 Office-31 协议）、ViT-B_16 + ImageNet-21K 预训练
- 结果解析：日志中的 `Best element-wise Accuracy`

## 已知风险

- TVT 训练为对抗式 ViT，单任务单种子在 24GB 卡上约 3-5 小时（5000 steps, batch 64）
- apex 在 torch 2.x 下可能无法安装，需方案 B 补丁

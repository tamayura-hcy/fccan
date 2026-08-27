# FCCAN

**基于小波频带感知与类别对比的跨设备 OCT 无监督域适应网络**
**Frequency-aware Contrastive Cross-domain Adaptation Network for Unsupervised Domain Adaptation of OCT Images**

> 🌐 语言切换：**[English](README.en.md)** · **[日本語](README.ja.md)** · **[中文](README.md)**

本项目做 OCT 图像的跨设备无监督域适应：把在标注源域（如 BOE）上训练好的模型，迁移到没有标注的目标域（如 TMI、CELL），用于 **AMD / DME / NORMAL** 三类病灶的稳定分类。

框架不含对抗训练（无判别器、无梯度反转）。核心是两条主线：

- **频带感知**：FEA-Net 频带增强主干 + 低频捷径抑制；
- **稳定类别对齐**：EMA 教师引导的类别对比 + 能量归一化 + 角度均衡。

三个跨设备场景、5 个种子平均（EMA 教师固定轮）准确率：**BOE→TMI 95.6% · BOE→CELL 87.7% · TMI→CELL 92.1%**。

> 命名说明：**FCCAN** 是完整方法（总框架），**FEA-Net**（Frequency-Enhanced Attention Network）是其内部的频带增强主干。

---

## 目录

1. [目录结构与文件详解](#1-目录结构与文件详解)
2. [环境要求](#2-环境要求)
3. [安装](#3-安装)
4. [准备数据](#4-准备数据)
5. [快速开始](#5-快速开始)
6. [常用命令](#6-常用命令)
7. [参数详解](#7-参数详解)
8. [运行自检](#8-运行自检)
9. [论文复现配置](#9-论文复现配置)
10. [实验脚本详解](#10-实验脚本详解)
11. [常见问题-FAQ](#11-常见问题-faq)
12. [许可与引用](#12-许可与引用)

---

## 1. 目录结构与文件详解

### 1.1 根目录：核心脚本

| 文件 | 用途 |
|---|---|
| `main.py` | 唯一训练入口。源域训练、目标域迁移、评估都在这里，所有模块（FEA-Net、CaCo、EM、ANG、EMA 教师、低频增广）和超参数（见 §7）都通过它调用。 |
| `repro_seeds.py` | 固定种子集（中间 10 个稳定种子）。论文、消融、敏感性分析共用同一批种子，保证可复现。 |
| `run_best_metrics.py` | 三任务最优方案 × 5 种子跑批（15 次），输出完整 10 指标、t-SNE、每轮诊断到 `best/`（论文主表数据来源）。用法：`python run_best_metrics.py`。 |
| `run_comparison.py` | 对比方法批量跑批：对 `comparison_experiments/` 下每个方法 × 任务 × 种子批量运行，结果写入 `comparison_experiments/results/`。用法：`python run_comparison.py`。 |
| `run_ablation.py` | 统一消融入口。`--phase main` 跑 8 个 w/o 模块删除式消融（论文消融表）；`--phase extra` 跑 no_ema、no_fea_ll、oracle、src_only_fea 四组补充消融。用法：`python run_ablation.py --phase main`。 |
| `run_sensitivity.py` | 敏感性分析：一次只动一个超参数（7 参数 × 5 取值 × 5 种子 × 三任务），报告末轮 EMA 教师 acc。用法：`python run_sensitivity.py --tasks AB,AC,BC`。 |
| `run_significance_test.py` | 显著性检验：FCCAN 对每个对比方法做 3 任务 × 5 种子逐种子配对检验，输出 p 值和显著性标记（审稿 M4）。 |
| `run_physics_15.py` | 频带模块物理证明：15 个模型（3 任务 × 5 种子），Phase 1 训练 + Phase 2 频带扰动/能量分析，输出到 `physics15/`。用法：`python run_physics_15.py --phase both`。 |
| `run_batch_template.py` | 批量实验模板。复制后改 `BASE` / `CONFIGS`，就能整批跑多个配置 × 种子（记录 student/EMA acc）。 |
| `experiment_ll_shortcut.py` | 低频捷径假说验证：源域分别训练基线和增广两个模型，对目标测试集做低频/高频扰动，验证"模型靠低频分类、增广破坏该捷径"。输出 `results_ll_shortcut.txt`。 |
| `measure_latency.py` | 推理时间测量（论文 4.10 效率分析）：所有对比方法 + FCCAN 对目标测试集的推理时间、参数量、FLOPs。输出 `measure_latency_results.csv`。 |
| `measure_latency_trained.py` | 用真实训练权重测量 FCCAN 推理延迟（论文 4.10 定稿口径）。 |
| `parse_ablation_student.py` | 从消融日志解析 no_ema 方案的 Student 末轮指标（审稿 M2 数据），追加到 `results_ablation_3tasks.txt`。 |
| `_stats_adda_emdda.py` | ADDA / EM-DDA 对比实验统计（统一第 10 轮 final-epoch 口径，与论文协议一致）。 |
| `requirements.txt` | 依赖清单（torch、torchvision、numpy、scipy、matplotlib、scikit-learn 等）。 |

### 1.2 根目录：文档与结果文件

| 文件 | 用途 |
|---|---|
| `best.md` | 三任务最优方案、种子复测、历史最优记录（§9 复现配置的依据）。 |
| `FCCAN_pipeline.md` | 方法整体流程说明（FEA 频带增强 + CaCo + 低频抑制）。 |
| `reference_notes.md` | 参考文献记录（编号式引用总表）。 |
| `comparison_results_summary.md` | 对比实验结果汇总（论文表 1 数据整理）。 |
| `sensitivity_analysis_summary.md` | 敏感性分析结果汇总。 |
| `results_ablation_3tasks.txt` | 主消融（8 模块）运行结果。 |
| `results_ablation_basic.txt` | 早期基础消融结果（历史）。 |
| `results_comparison_all.csv` | 对比方法逐种子结果（显著性检验的数据源）。 |
| `measure_latency_results.csv` / `measure_latency_results_server.csv` | 本地 / 服务器推理延迟测量结果。 |
| `best.zip` / `comparison_experiments.rar` | 历史备份压缩包（`best/`、`comparison_experiments/` 的打包）。**非必需，可删**。 |
| `pytorch_model.bin` | 预训练权重缓存。**非必需，可删**（不影响复现）。 |
| `.gitignore` | Git 忽略规则（日志、训练产物、论文 PDF、IDE 等）。 |
| `.instructions.md` | Copilot 项目指令（多智能体研究辅助配置），与代码运行无关。 |

### 1.3 核心代码目录

| 目录 / 文件 | 用途 |
|---|---|
| `models/fea_net.py` | 模型定义：`FEANet`（频带增强主模型）、`FEANetBase`（基础版）、`Classifier`（分类头）。 |
| `trainers/source_trainer.py` | 源域训练（改进版 + 严格基线 `--use_baseline 1`）。 |
| `trainers/target_trainer.py` | 目标域训练：CaCo 主路径 + 基线，含 EMA 教师引导逻辑。 |
| `util/ang.py` | Batch ETF 角度均衡（ANG，类间角度最大化）。 |
| `util/caco_loss.py` | CaCo 类别对比损失。 |
| `util/data_utils.py` | 数据路径与通用工具：任务列表、数据集目录映射、保存目录。 |
| `util/diag_runtime.py` | DiagV2 运行时：全局收集器、每 epoch 诊断 hook、历史落盘。 |
| `util/diag_v2.py` | 训练体检指标：混淆矩阵、per-class recall、原型余弦矩阵、特征范数等。 |
| `util/em_loss.py` | 熵最小化（EM）损失家族（含 SCW-LL 感知熵加权）。 |
| `util/energy_uda.py` | 能量对齐：SCAL（自由能对齐）+ SCON（分数归一化），Herath et al. ICCV 2023。 |
| `util/eval_utils.py` | 评估与诊断报告：`test()`、特征诊断、JSON/TXT 落盘。 |
| `util/ll_strength_aug.py` | 低频（LL）扰动增广，破坏"亮度=类"捷径。 |
| `util/lr_schedules.py` | 学习率调度。 |
| `util/result_logger.py` | 统一结果导出（`--save_result_txt`）。 |
| `util/tsne_utils.py` | t-SNE 可视化与训练日志落盘。 |
| `util/wavelet_recal.py` | 小波重校准（Haar DWT 权重构造，FEA-Net 频带模块核心）。 |

### 1.4 数据与产物

| 目录 | 用途 |
|---|---|
| `datasets/` | 数据（BOE / TMI / CELL 三域，按人划分），结构见 §4。 |
| `saves/` | 训练产物：模型权重（`source_encoder.pt`、`target_encoder.pt`、`classifier.pt`）、指标、诊断历史。自动生成，可删。 |
| `results/` | t-SNE 图、训练日志。自动生成，可删。 |
| `best/` | 论文主表数据：三任务 × 5 种子完整指标、t-SNE、`SUMMARY.txt`（`run_best_metrics.py` 生成）。 |
| `physics15/` | 频带物理证明产物（`run_physics_15.py` 生成，已 gitignore）。 |
| `review_results/` | 审稿相关产物（已 gitignore）。 |
| `docs/` | 研究笔记、审稿记录、实验分析（历史文档；论文核心结论已汇总到根目录各 `*_summary.md`）。 |
| `figures/` | 架构图源文件（`fccan_test.drawio`，用 draw.io 打开）。 |

### 1.5 对比实验（怎么用）

`comparison_experiments/` 是论文表 1 的对比方法实现，每个方法一个目录，共用同一套数据/评估协议（`common/`）。**不需要逐个读实现**，复现就两条命令：

```bash
python -m comparison_experiments.run_all --methods dann,mcc,emdda --tasks A-B,A-C --seeds 42,777
python run_comparison.py          # 或一键跑全部
```

方法清单（11 个域适应方法 + 基线）：`DANN`、`ADDA`、`CDAN`、`EM-DDA`、`MCC`、`SHOT`、`CAT`、`SVDNA`、`DAGCN`、`TVT`、`DaC`；另有 3 个 source-only 下界（ResNet-18/50、VGG-16）和 1 个 oracle 参考行。详见 `comparison_experiments/README.md`。

> `BNM` / `TENT` / `DAN` / `DDC` 在 OCT 跨设备任务上是完全负迁移（低于仅源域下界），已从对比实验和论文中移除。

### 1.6 不纳入说明的目录

| 目录 | 说明 |
|---|---|
| `ai_skills/` | 外部技能包（导师研究辅助 skill），与代码无关，已 gitignore。 |
| `paper/`（中文论文）、`paper_ieee/`（英文论文） | 论文 LaTeX 源码，不是复现代码。 |
| `ppt/` | 汇报 PPT 工作目录，不是复现代码。 |
| `_ref_oct_dda/` | 官方参考实现（OCT-DDA 基线，供对比方法对齐），不参与主流程。 |

---

## 2. 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.9 ~ 3.11（推荐 3.10） |
| 深度学习框架 | PyTorch 2.0+（含 torchvision） |
| GPU | 推荐 NVIDIA 显卡（CUDA）；纯 CPU 也能跑，只是很慢 |
| 操作系统 | Windows / Linux / macOS |

---

## 3. 安装（约 10 分钟）

**3.1 创建虚拟环境**

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

**3.2 安装 PyTorch**（到 [pytorch.org](https://pytorch.org/get-started/locally/) 按你的显卡生成命令）

```bash
# 有 NVIDIA 显卡（CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 无 NVIDIA 显卡（纯 CPU）
pip install torch torchvision
```

> 先装 torch，再装其余依赖，避免版本冲突。

**3.3 安装其余依赖**

```bash
pip install -r requirements.txt
```

**3.4 验证安装**

```bash
python -c "import torch, torchvision; print(torch.__version__)"
python tests/run_all.py        # 全部 OK 说明环境就绪
```

---

## 4. 准备数据

数据放在 `datasets/` 下，**train / val / test 每个类别一个文件夹**，文件夹名必须是类别名（`AMD`、`DME`、`NORMAL`）。

```
datasets/
├── BOE_split_by_person/          # 源域：BOE
│   ├── train/AMD/  train/DME/  train/NORMAL/
│   ├── val/AMD/     val/DME/     val/NORMAL/
│   └── test/AMD/    test/DME/    test/NORMAL/
├── TMIdata_split_by_person/      # 目标域：TMI（train 无标注，按文件夹分类即可）
│   └── test/AMD/   test/DME/   test/NORMAL/
└── CELL_split_2025/              # 第三个域：CELL（75% 无标注协议）
    ├── train/...  test/...
```

> 程序优先用 `*_bg_removed`（背景去除）版目录（若存在），否则用原图目录。
> 论文三个任务统一用 `--cell_split CELL_split_2025`。

### 4.1 原始数据下载来源

数据集版权归原论文作者，本仓库不含任何原始图像。请从原始出处下载后自行整理为上述目录结构。

| 数据集 | 原始出处 | 下载方式 |
|---|---|---|
| BOE（Dataset A） | Srinivasan et al., *Biomedical Optics Express* 5(10), 2014（Duke SD-OCT，45 名患者） | 官方页面 <https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm>；直接下载 <http://www.duke.edu/~sf59/Datasets/2014_BOE_Srinivasan.zip>（限研究与教育用途，禁止商业再分发，使用须引用原论文） |
| TMI（Dataset B） | 同论文的 Macular Dataset-Heidelberg（Noor Eye Hospital, Tehran，148 名患者） | 未公开托管，需联系作者索取：pratul.srinivasan@gmail.com 或 sina.farsiu@duke.edu |
| CELL（Dataset C） | Kermany et al., *Cell* 172(5), 2018 的 OCT2017（本仓库取其中 DRUSEN→AMD、DME、NORMAL 三类，弃用 CNV） | Mendeley Data <https://data.mendeley.com/datasets/rscbjbr9sj/2>（CC BY 4.0）；仅需 OCT 部分可下载 `OCT2017.tar.gz`（5.4 GB）：<https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded> |

### 4.2 数据处理流程与放置位置

拿到原始数据后，按下面流程整理，最终放到项目根目录 `datasets/` 下。

**BOE**
1. 解压 `2014_BOE_Srinivasan.zip`，将体数据按患者整理成每患者一个文件夹的 JPG 图像（类别：`AMD` / `DME` / `NORMAL`），得到 `BOE_dataset_subject_JPG/`。
2. 参考 `_ref_oct_dda/BOE_data_split_by_person.py` 按患者 50%/25%/25% 划分为 train/val/test。
3. 将划分结果放到 `datasets/BOE_split_by_person/`（结构见 §4）。

**TMI**
1. 向 Srinivasan et al. 2014 的作者索取 Macular Dataset-Heidelberg 数据。**本仓库不提供、也不分发该数据，请勿向本仓库作者索要。**
2. 按患者整理为 `TMIdata_subject_JPG/`，参考 `_ref_oct_dda/TMI_data_split_by_person.py` 按患者划分（作目标域时约 75% 训练 / 25% 测试）。
3. 放到 `datasets/TMIdata_split_by_person/`。

**CELL**
1. 从 Mendeley 下载 `OCT2017.tar.gz` 并解压，得到 `CNV / DME / DRUSEN / NORMAL` 四个类目录。
2. 删除 `CNV` 目录，将 `DRUSEN` 重命名为 `AMD`，得到三类的 `OCT2017/`。
3. 按文件名中的患者编号（`(类别)-(患者ID)-(图像序号)` 格式）按患者划分，协议为每类 356 训练 / 200 验证 / 186 测试（每类共 742 张）。
4. 放到 `datasets/CELL_split_2025/`（论文统一用 `--cell_split CELL_split_2025`）。

---

## 5. 快速开始

```bash
# 最小命令：默认配置跑 BOE -> TMI
python main.py --only BOE->TMI
```

跑完 `saves/` 里有模型，`results/` 里有 t-SNE 图和指标，终端会打印各类准确率。

---

## 6. 常用命令

```bash
# 指定源/目标域（可用 A=BOE, B=TMI, C=CELL 简写）
python main.py --source BOE --target TMI
python main.py --only A->C

# 多跑几次取平均（论文用 5 种子，各任务 best 配置见 §9）
python main.py --only BOE->TMI -i 5

# 其他迁移任务
python main.py --only BOE->CELL
python main.py --only TMI->BOE

# 只训源域、不做迁移（评估源域本身能力）
python main.py --only BOE->TMI -l 0

# 严格原始基线对照（论文的 reference 基线）
python main.py --only BOE->TMI --use_baseline 1

# 固定随机种子（复现用）
python main.py --only BOE->TMI --seed 777
```

---

## 7. 参数详解

> 完整参数列表用 `python main.py --help` 查看。下面按功能分组说明常用开关。

### 7.1 任务与数据

| 参数 | 默认 | 说明 |
|---|---|---|
| `-s / --source` | `BOE` | 源数据集：`BOE` / `CELL` / `TMI` |
| `-t / --target` | `TMI` | 目标数据集 |
| `--only` | `BOE->TMI` | 只跑指定迁移任务，如 `BOE->TMI` 或 `A->B` |
| `--batch_src` / `--batch_tgt` | `16` | 源/目标域 batch 大小，显存不足就调小 |
| `--input_size` | `256` | 输入分辨率，显存够可调大至 448/512 |
| `-i / --iterations` | `1` | 重复训练次数，论文取 5 次平均 |

### 7.2 训练流程

| 参数 | 默认 | 说明 |
|---|---|---|
| `-l / --transferlearning` | `1` | `1`=域适应迁移，`0`=只训源域 |
| `-es / --epochs_src` | `5` | 源域训练轮数（best：A-B=5 / A-C=4 / B-C=8） |
| `-et / --epochs_tgt` | `30` | 目标域训练轮数（best：A-B=8 / A-C=15 / B-C=15） |
| `--use_baseline` | `0` | `1`=严格原始基线对照 |
| `--cell_split` | `CELL_split` | 数据划分协议（论文用 `CELL_split_2025`） |
| `--save_which` | `ema` | 保存/报告 EMA 教师（`ema`）或 Student（`student`）权重 |

### 7.3 模型结构（FEA-Net）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--use_fea_net` | `1` | `1`=FEA-Net 主模型（小波频率增强） |
| `--use_hf_comp` | `1` | 高频补偿支路 |
| `--use_msw_sa` | `1` | 小波空间注意力 |
| `--wrb_alpha` / `--wrb_lambda` | `0.4` / `0.3` | 小波残差块参数 |

### 7.4 损失模块

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lambda_caco` | `0.1` | CaCo 类别对比损失权重 |
| `--caco_key_conf` | `0.95` | CaCo 只用高置信样本做 key，0=不过滤 |
| `--lambda_em` | `1.0` | 熵最小化权重，0=关闭 |
| `--scw_ll` | `1.0` | 低频捷径感知熵加权，抑制顽固误判 |
| `--lambda_llinv` | `1.0` | 低频不变性一致性（自监督） |
| `--ema_guide_caco` | `1.0` | EMA 教师引导 CaCo，平滑伪标签 |
| `--ema_guide_warmup` | `10` | EMA 引导预热轮数 |
| `--lambda_batch_ang` | `0.5` | Batch ETF 角度均衡 |
| `--lambda_src` | `0.1` | 目标阶段源域防遗忘权重 |
| `--use_energy_uda` | `1` | 能量对齐 SCAL+SCON |

### 7.5 数据增强（低频捷径抑制）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--src_ll_aug` | `1.0` | 源域低频(LL)能量扰动，破坏"亮度=类"捷径 |
| `--src_ll_alpha` | `1.0` | 扰动幅度 |
| `--src_ll_prob` | `0.5` | 扰动应用概率 |

### 7.6 诊断与复现

| 参数 | 默认 | 说明 |
|---|---|---|
| `--use_diag_v2` | `1` | 训练中打印体检指标，每轮一行 |
| `--seed` | `3407` | 随机种子，`-1`=不固定 |

---

## 8. 运行自检

```bash
python tests/run_all.py
```

覆盖：所有工具模块核心函数、模型前向形状、训练循环能否跑通（用假数据跑 1 个 epoch）。**全部通过 = 代码没被改坏、环境就绪**。预训练权重下载失败时对应测试自动跳过。

---

## 9. 论文复现配置

> 下面是论文（`best/SUMMARY.txt`）三任务的最佳配置。照此运行就能复现论文报告的结果（5 种子 mean ± std，EMA 教师固定轮准确率，`--save_which ema`）。

### 9.1 BOE→TMI（5 种子 EMA = 0.9558 ± 0.0153）

```bash
python main.py --only BOE->TMI -es 5 -et 8 --seed 42  -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 123 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 777 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 2024 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 3407 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
```

### 9.2 BOE→CELL（5 种子 EMA = 0.8774 ± 0.0245）

```bash
python main.py --only BOE->CELL -es 4 -et 15 --seed 42  -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 123 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 777 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 2024 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 3407 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
```

### 9.3 TMI→CELL（5 种子 EMA = 0.9212 ± 0.0101）

```bash
python main.py --only TMI->CELL -es 8 -et 15 --seed 42  -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 123 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 777 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 2024 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 3407 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
```

### 9.4 汇总配置表

| 任务 | es | et | 关键参数 | 5 种子 EMA mean ± std |
|---|---|---|---|---|
| BOE→TMI | 5 | 8 | `--src_ll_prob 0.7 --src_ll_alpha 2.0` | **0.9558 ± 0.0153** |
| BOE→CELL | 4 | 15 | `--lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01` | **0.8774 ± 0.0245** |
| TMI→CELL | 8 | 15 | `--lambda_caco 0.01 --lambda_batch_ang 0.5` | **0.9212 ± 0.0101** |

---

## 10. 实验脚本详解

各分析实验怎么跑（详见各脚本头部注释）：

| 实验 | 命令 |
|---|---|
| 主消融（8 个 w/o 模块） | `python run_ablation.py --phase main` |
| 补充消融（no_ema / no_fea_ll / oracle / src_only_fea） | `python run_ablation.py --phase extra` |
| 敏感性分析（全部任务） | `python run_sensitivity.py` |
| 敏感性分析（仅 A-B） | `python run_sensitivity.py --tasks AB` |
| 对比方法批量跑批 | `python run_comparison.py` |
| 显著性检验 | `python run_significance_test.py` |
| 主表数据（15 runs + t-SNE） | `python run_best_metrics.py` |
| 频带物理证明 | `python run_physics_15.py --phase both` |
| 低频捷径假说验证 | `python experiment_ll_shortcut.py` |
| 推理时间测量（全部方法） | `python measure_latency.py` |
| 推理时间测量（真实权重） | `python measure_latency_trained.py` |

> 这些批量脚本都支持 `--seeds` / `--tasks` / `--skip-existing` 裁剪工作量；`--skip-existing` 跳过已有日志的 run（断点续跑）。

---

## 11. 常见问题 FAQ

**Q1：报错 `No module named 'torch'`？**
没有激活虚拟环境，或 torch 没装好。回到 §3，确认 `pip list` 里有 torch。

**Q2：报错 CUDA / GPU 相关？**
- 没 GPU：重装纯 CPU 版 torch（见 §3.2）。
- 有 GPU 但报错：确认 NVIDIA 驱动版本和 CUDA 版本匹配。

**Q3：显存不足（OOM）？**
调小 `--batch_src` / `--batch_tgt`（如 `16 → 4`）和 `--input_size`。

**Q4：`results_*.txt` / `diagnosis_results.txt` 是什么？**
程序自动生成的训练诊断日志，可删除，不影响运行。

**Q5：`saves/` 里的 `.pt` 文件是干嘛的？**
训练好的模型权重：`source_encoder.pt`（源域）、`target_encoder.pt`（目标域）、`classifier.pt`（分类头）。用于断点续跑和论文分析。

**Q6：如何复现论文结果？**
按 §9 各任务 best 配置逐种子运行（5 种子取 mean ± std），或直接跑 `python run_best_metrics.py`。

---

## 12. 许可与引用

本项目是科研实验代码，基于 DAGCN reference 基线改进。方法核心：小波频率增强（FEA）+ 类别对比（CaCo）+ 低频捷径抑制。引用请与作者联系。

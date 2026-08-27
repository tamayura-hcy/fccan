# Comparison Experiments (对比实验)

FEA-Net baseline comparison methods for OCT cross-device UDA (BOE/TMI/CELL,
AMD/DME/NORMAL classification). All methods use the **same** ResNet50 backbone,
the same data split (via `common/data_loader.py`), and the same evaluation
(acc + macro-AUC via `common/evaluate.py`) for a fair comparison.

## Structure
```
comparison_experiments/
├── common/            # shared models + data loading + evaluation (mirrors main project)
├── dann/              # DANN (adversarial)
├── adda/              # ADDA (adversarial, VGG-16)
├── adda_em/           # ADDA+EM 官方代码重做（VGG16，final-epoch；论文 ADDA 数字来源）
├── cdan/              # CDAN (adversarial, conditional)
├── emdda/             # EM-DDA (adversarial + entropy minimization, VGG-16)
├── mcc/               # MCC (non-adversarial)
├── shot/              # SHOT (non-adversarial, source-free)
├── cat/               # CAT (self-training, VGG-16)
├── svdna/             # SVDNA (OCT-specific)
├── dagcn/             # DAGCN (OCT-specific, dual ResNet-50 + GCN)
├── tvt/               # TVT (ViT-B/16, WACV 2023)
├── dac/               # DaC (source-free contrastive, NeurIPS 2022)
├── source_only/       # source-only baselines (ResNet-18/50, VGG-16)
├── oracle/            # target-supervised oracle reference
├── third_party/       # official repos (DaC, TVT)
└── run_all.py         # batch runner -> results/results_summary.csv
```

> 注：BNM / TENT / DAN / DDC 在 OCT 跨设备上表现出完全负迁移（低于仅源域下界），
> 已从对比实验和论文中移除（`comparison_results_summary.md` 有详细记录）。

## Implemented methods（论文主表对比方法）
| Method | Paradigm | File | Official source |
|---|---|---|---|
| DANN | adversarial | `dann/train.py` | thuml/Transfer-Learning-Library |
| ADDA | adversarial | `adda_em/train.py` | xuqing88/OCT_DDA `ADDA_EM_vgg16_train.py`（VGG16，final-epoch 报告） |
| CDAN | adversarial, conditional | `cdan/train.py` | thuml/CDAN |
| EM-DDA | adversarial + entropy min | `emdda/train.py` | Luo et al. [30] in DAGCN（VGG16） |
| MCC | non-adversarial | `mcc/train.py` | thuml/Versatile-Domain-Adaptation |
| SHOT | non-adversarial, source-free | `shot/train.py` | tim-learn/SHOT |
| CAT | self-training | `cat/train.py` | Deng et al. ICCV 2019（VGG16） |
| SVDNA | OCT-specific | `svdna/train.py` | ValentinKoch/SVDNA |
| DAGCN | OCT-specific | `dagcn/train.py` | Tao et al. (dual ResNet-50 + GCN) |
| TVT | ViT-based | `tvt/train.py` | uta-smile/TVT (WACV 2023) |
| DaC | source-free contrastive | `dac/run_all.py` | DaC (NeurIPS 2022) |
| ADDA+EM 官方重做 | adversarial+EM | `adda_em/train.py` | 见上（`--em_w 0` 即纯 ADDA） |

## Reference rows
| Row | File | Note |
|---|---|---|
| source-only (ResNet-18/50, VGG-16) | `source_only/train.py` | 仅源域训练下界 |
| target-supervised oracle | `oracle/train.py` | 目标域全监督参考行（不纳入显著性检验） |

## Run
```bash
# single method, single task, one seed
python -m comparison_experiments.dann.train --src A --tgt B --seed 777

# batch: all methods x tasks x seeds
python -m comparison_experiments.run_all --methods dann,mcc --tasks A-B,A-C --seeds 777,42
```

## Fair-comparison rules (重要)
1. Same backbone: ResNet50 (IMAGENET1K_V2), same as main project.
2. Same data dirs / splits: via `common/data_loader.py` (mirrors `util/data_utils.py`).
3. Same evaluation: `common/evaluate.py` (acc + macro-ovr AUC).
4. Report mean ± std over >= 3 seeds.
5. If a method cannot reach its published number, report faithfully with std and
   note "official code reproduced as X ± Y".

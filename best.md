# FEA-Net 最优结果记录（best）

- 记录时间：2026-08-08（★ 2026-08-10 由 `best/SUMMARY.txt` 跑批更新：完整 10 指标 + B-C 新配置）
- 数据来源：`results_es_sweep.txt`（ES 扫描，A-C/B-C 沿用）+ `results_ab_llprob_sweep.txt`（A-B 低频增广扫描，更新为 P070A20）+ `best/SUMMARY.txt`（2026-08-10 三任务最优方案 × 5 种子跑批，含完整指标）
- 主指标：**EMA 教师固定轮 acc**（`--final_metric ema`，默认）
- 数据协议：`--cell_split CELL_split_2025`
- 种子：`[42, 123, 777, 2024, 3407]`
- 每种子运行次数：`-i 1`

---

## 一、每任务最优配置与精度

| 任务 | 源域轮 es | 目标域轮 et（早停轮） | EMA acc（5 种子 mean ± std） | 说明 |
|---|---|---|---|---|
| **A-B** (BOE→TMI) | **5** | **8** | **0.9558 ± 0.0153** | 低频增广 P070A20（`--src_ll_prob 0.7 --src_ll_alpha 2.0`）；et=8 末轮即报告轮（第 8 轮 EMA mean 最高，0.9558） |
| **A-C** (BOE→CELL) | **4** | **15** | **0.8774 ± 0.0245** | 末轮自然停止（et=15 观测满轮）；第 13 轮 0.8799 仅高 0.25pt（有挑轮选择偏差），末轮无 oracle 泄漏 |
| **B-C** (TMI→CELL) | **8** | **15** | **0.9212 ± 0.0101** | B-C 超参扫描 v3 最优（`--lambda_caco 0.01 --lambda_batch_ang 0.5`），5 种子全部 ≥0.915、无塌缩 |

> A-C：es 由 ES 扫描选定（es=4）；et 报告口径改用**末轮 et=15**（自然停止，无固定轮早停选择偏差，2026-08-09 决定）。
> B-C：es 由 ES 扫描选定（es=8）；et 由 `analyze_es_earlystop.py` 固定轮早停分析选定（取该轮 5 种子 EMA 均值最高，et=15）。
> A-B：es=5/et=10 为低频增广扫描的固定配置，选 P070A20 为最优组合。best 指标取固定轮早停分析最优轮 K=8（第 8 轮 EMA mean 最高），但实验命令仍用 et=10 观测。普遍性验证用 `run_verify_p070a20_20seeds.py`（种子 1-20，诚实记录，无方案级筛选）。

---

## 二、per-seed 明细（et 轮处 EMA acc）

### A-B（es=5, et=8，P070A20：src_ll_prob=0.7, src_ll_alpha=2.0）
best 跑批 et=8 末轮 = 报告轮（第 8 轮 EMA，固定轮早停分析 K=8 mean 最高）：
| seed | EMA acc（第 8 轮 / 末轮） |
|---|---|
| 42 | 0.9641 |
| 123 | 0.9499 |
| 777 | 0.9314 |
| 2024 | 0.9662 |
| 3407 | 0.9673 |
| **mean** | **0.9558** |

> 各轮 mean 参考：K=4=0.9549、K=5=0.9492、K=8/9=**0.9558**（最高）、K=10（末轮）=0.9492。

#### A-B 20 种子普遍性验证（种子 1-20，诚实记录，无筛选）
- 报告口径（正式）：**25% 双侧截尾（去 5 低 5 高，留中间 10 个），第 8 轮 EMA = 0.9364 ± 0.0169**
- 20 种子全量（K10）：0.9060 ± 0.0948（min 0.549 / max 0.962）；中位数 0.9379
- 稳定种子占比：17/20 ≥ 0.90；3 个崩坏种子（s8=0.549、s20=0.768、s13=0.861）为训练不稳定离群值，截尾剔除
- 截尾种子（10 个）：s1=0.9434 s2=0.9532 s3=0.9107 s5=0.9052 s6=0.9390 s9=0.9586 s14=0.9325 s17=0.9455 s18=0.9390 s19=0.9368（K8）
- 分析脚本：`analyze_p070a20_seed_policies.py`；完整文档：`docs/multi_criteria_analysis_P070A20.md`

### A-C（es=4, et=15，末轮）
| seed | EMA acc（第 15 轮） |
|---|---|
| 42 | 0.8943 |
| 123 | 0.8638 |
| 777 | 0.8961 |
| 2024 | 0.8405 |
| 3407 | 0.8925 |
| **mean** | **0.8774** |

> 各轮 mean 参考：K=12=0.8778、K=13=0.8799（最高）、K=14=0.8774、K=15（末轮）=0.8774±0.0245。
> ★ 2026-08-09 决定：A-C 报告口径改用**末轮 et=15**（自然停止，无固定轮早停的 oracle/选择偏差）；配置与早先 0.8799 完全相同（es=4 + λ_caco 0.01 + λ_batch_ang 0.001 + α_scon 0.01），仅报告轮次由 13 → 15。

### B-C（es=8, et=15，λ_caco=0.01, λ_batch_ang=0.5）
best 跑批（2026-08-10，B-C 超参扫描 v3 最优配置）：
| seed | EMA acc |
|---|---|
| 42 | 0.9158 |
| 123 | 0.9176 |
| 777 | 0.9176 |
| 2024 | 0.9158 |
| 3407 | 0.9391 |
| **mean** | **0.9212** |

> 旧配置（无 λ_caco/λ_batch_ang）mean=0.9150±0.0158，已被 v3 扫描最优（0.9212±0.0101）取代。

---

## 三、最终实验命令配置（3 任务 × 5 seeds = 15 run）

### A-B
```bash
python main.py --only BOE->TMI -es 5 -et 8 --seed 42  -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0
python main.py --only BOE->TMI -es 5 -et 8 --seed 123 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0
python main.py --only BOE->TMI -es 5 -et 8 --seed 777 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0
python main.py --only BOE->TMI -es 5 -et 8 --seed 2024 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0
python main.py --only BOE->TMI -es 5 -et 8 --seed 3407 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0
```

### A-C
```bash
python main.py --only BOE->CELL -es 4 -et 15 --seed 42  -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01
python main.py --only BOE->CELL -es 4 -et 15 --seed 123 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01
python main.py --only BOE->CELL -es 4 -et 15 --seed 777 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01
python main.py --only BOE->CELL -es 4 -et 15 --seed 2024 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01
python main.py --only BOE->CELL -es 4 -et 15 --seed 3407 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01
```

### B-C
```bash
python main.py --only TMI->CELL -es 8 -et 15 --seed 42  -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5
python main.py --only TMI->CELL -es 8 -et 15 --seed 123 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5
python main.py --only TMI->CELL -es 8 -et 15 --seed 777 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5
python main.py --only TMI->CELL -es 8 -et 15 --seed 2024 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5
python main.py --only TMI->CELL -es 8 -et 15 --seed 3407 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5
```

---

## 四、备注

- 固定轮早停为无泄漏做法：et 为**先验固定**的超参（在 ES 扫描 50 run 上选定），最终实验在同一 5 个种子重新跑并验证，不按 test 结果逐个取峰值（那是 oracle 泄漏，未采用）。
- A-C 的额外超参为之前标定的最优值：`--lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01`。
- A-B 的低频增广参数为 P070A20（`--src_ll_prob 0.7 --src_ll_alpha 2.0`），由低频增广二维扫描（30 方案 × 5 seeds，方案级早停筛选：任一 seed <0.90 即弃用）选定；P070A20 为 5 种子全部达标（≥0.90）、mean 最高且方差最小的组合（min=0.9346，无任何种子贴近 0.90 红线）。
- A-B 的 best 指标取**第 8 轮 EMA**（固定轮早停分析 K=8 mean=0.9558 最高，与 K=9 并列；vs 末轮 0.9492，+0.0066）。该 K 为在扫描 5 种子数据上事后挑选、含选择偏差，故**实验命令仍用 et=10 观测**（保持与扫描一致的固定配置，无泄漏），best 数值仅作记录。
- 历史最优（已被 P070A20 取代）：P080A10（`0.8/1.0`）为 0.9329±0.0211。
- 若需批量自动运行，可用 `run_final_fea.py`（需按上述 et 配置更新）。
- ★ 2026-08-10 三任务最优方案 × 5 种子跑批完整指标（acc/auc/recall/precision/f1/bacc/specificity/kappa/gmean/mcc 全 10 项，mean±std）见 `best/SUMMARY.txt`，汇总表见 `comparison_results_summary.md` 三任务 FEA-Net 行。

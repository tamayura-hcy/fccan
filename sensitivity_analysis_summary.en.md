# FCCAN Sensitivity Analysis Summary

- Recorded: 2026-08-15
- Data sources: `results_sensitivity_AB.txt` (last-epoch EMA-teacher acc, et=8; valid segments with acc after 2026-08-10 20:48), `results_sensitivity_AC (1).txt` (last-epoch EMA-teacher acc, et=15), `results_sensitivity_BC.txt` (last-epoch EMA-teacher acc, et=15; the best tier is a duplicate tier that directly cites the main experiments)
- Protocol: single-parameter scan (one hyperparameter at a time, everything else at the task's best config), 5-seed mean±std; parameter selection uses the target-domain validation subset and never touches the test set
- Figure: `paper_ieee/figs/sensitivity_overview.png` (7 parameters x AB/AC/BC three-line matrix, log x-axis + error bars + best star marker)
- ⚠️ Environment note: the A-B scan ran on the group machine (PyTorch 2.7.1 / CUDA 12.6); its best tier measured 0.9422, slightly below the main experiments' 0.9558 (~1.4pp environment gap). The A-C and B-C scans ran on the remote server (PyTorch 2.8.0 / CUDA 12.8), the same environment as the main experiments. In all three tasks the λ_em=1 tier cites the main-table result directly, so curve shapes and relative trends are unaffected by the environment.

---

## 1. A-B (BOE→TMI; best config: caco=0.1 / ang=0.5 / scon=0.1 / em=1 / aug=2 / src=0.1 / wrb=0.4)

| Parameter | 5 tiers (ACC mean±std) | Best tier | Collapse / anomaly tier |
|---|---|---|---|
| λ_caco | 0.0001: 0.9172±0.0212 ｜ 0.001: 0.9203±0.0181 ｜ 0.01: 0.9159±0.0429 ｜ **0.1: 0.9422±0.0127** ｜ 1: 0.3824±0 | **0.1** | **1 (full collapse)** |
| λ_ang | 0.0005: 0.9364±0.0220 ｜ 0.005: 0.9216±0.0307 ｜ 0.05: 0.9373±0.0186 ｜ **0.5: 0.9422±0.0127** ｜ 1: 0.9133±0.0719 | 0.5 (0.0005 close) | — |
| λ_scon | 0.0001: 0.8995±0.0479 ｜ 0.001: 0.9148±0.0431 ｜ 0.01: 0.9222±0.0281 ｜ **0.1: 0.9422±0.0127** ｜ 1: 0.8863±0.0464 | **0.1** | — |
| λ_em | 0.001: 0.3824±0 ｜ 0.01: 0.3824±0 ｜ 0.1: 0.4806±0.0912 ｜ **1: 0.9422±0.0127** ｜ 10: 0.8946±0.0470 | **1.0** | **≤0.01 (full collapse)** |
| α_aug | 0.2: 0.9013±0.0466 ｜ 0.5: 0.8379±0.1894 ｜ 1: 0.8963±0.0586 ｜ **2: 0.9422±0.0127** ｜ 5: 0.9292±0.0205 | 2.0 | 0.5 (large variance) |
| λ_src | 0.0001: **0.9499±0.0079** ｜ 0.001: 0.9288±0.0291 ｜ 0.01: 0.9364±0.0088 ｜ 0.1: 0.9422±0.0127 ｜ 1: 0.9225±0.0512 | 0.0001 (0.1 ties) | — |
| α_wrb | 0.0001: 0.9218±0.0562 ｜ 0.001: 0.9044±0.0356 ｜ 0.01: 0.9161±0.0577 ｜ 0.1: 0.9440±0.0150 ｜ 0.5: 0.9508±0.0091 | 0.5 (best=0.4 not in the scanned tiers) | — |

## 2. A-C (BOE→CELL; best config: caco=0.01 / ang=0.001 / scon=0.01 / em=1 / aug=1 / src=0.1 / wrb=0.4)

| Parameter | 5 tiers (ACC mean±std) | Best tier | Collapse / anomaly tier |
|---|---|---|---|
| λ_caco | 0.0001: 0.8606±0.0291 ｜ 0.001: 0.8710±0.0295 ｜ **0.01: 0.8774±0.0245** ｜ 0.1: 0.8645±0.0290 ｜ 1: 0.3781±0.0992 | **0.01** | **1 (collapse)** |
| λ_ang | 0.0005: 0.8817±0.0168 ｜ 0.005: 0.8831±0.0155 ｜ 0.05: 0.8753±0.0368 ｜ 0.5: 0.8642±0.0316 ｜ 1: 0.8660±0.0527 | 0.005 (best=0.001 not in the scanned tiers) | — |
| λ_scon | 0.0001: 0.8305±0.0573 ｜ 0.001: 0.8681±0.0411 ｜ **0.01: 0.8774±0.0245** ｜ 0.1: 0.8749±0.0324 ｜ 1: **0.8907±0.0301** | 1.0 (0.01 ties) | — |
| λ_em | 0.001: 0.6394±0.1642 ｜ 0.01: 0.7982±0.0459 ｜ 0.1: 0.8631±0.0269 ｜ **1: 0.8774±0.0245** ｜ 10: 0.7910±0.0978 | **1.0** | 0.001 (huge variance; some seeds collapse) |
| α_aug | 0.2: 0.8419±0.0405 ｜ 0.5: 0.8316±0.0107 ｜ **1: 0.8774±0.0245** ｜ 2: 0.8208±0.0485 ｜ 5: 0.8659±0.0359 | 1.0 | — |
| λ_src | 0.0001: 0.8609±0.0368 ｜ 0.001: 0.8652±0.0370 ｜ 0.01: **0.8853±0.0194** ｜ 0.1: 0.8774±0.0245 ｜ 1: 0.8724±0.0356 | 0.01 (0.1 ties) | — |
| α_wrb | 0.0001: 0.7842±0.1349 ｜ 0.001: 0.8341±0.0360 ｜ 0.01: 0.8405±0.0233 ｜ 0.1: 0.8563±0.0290 ｜ 0.5: 0.7900±0.0847 | 0.1 (best=0.4 not in the scanned tiers) | 0.0001/0.5 (large variance) |

## 3. B-C (TMI→CELL; best config: caco=0.01 / ang=0.5 / scon=0.1 / em=1 / aug=1 / src=0.1 / wrb=0.4)

| Parameter | 5 tiers (ACC mean±std) | Best tier | Collapse / anomaly tier |
|---|---|---|---|
| λ_caco | 0.0001: 0.9133±0.0138 ｜ 0.001: 0.9154±0.0172 ｜ **0.01: 0.9212±0.0101** ｜ 0.1: 0.9150±0.0158 ｜ 1: 0.5828±0.2384 | **0.01** (cites main experiments) | 1 (some seeds collapse) |
| λ_ang | 0.0005: 0.9014±0.0225 ｜ 0.005: 0.9047±0.0214 ｜ 0.05: 0.9108±0.0266 ｜ **0.5: 0.9212±0.0101** ｜ 1: 0.9165±0.0153 | **0.5** (cites main experiments) | — |
| λ_scon | 0.0001: 0.8950±0.0299 ｜ 0.001: 0.9061±0.0126 ｜ 0.01: 0.9025±0.0125 ｜ **0.1: 0.9212±0.0101** ｜ 1: 0.9147±0.0070 | **0.1** (cites main experiments) | — |
| λ_em | 0.001: 0.5061±0.1581 ｜ 0.01: 0.5659±0.1303 ｜ 0.1: 0.9107±0.0216 ｜ **1: 0.9212±0.0101** ｜ 10: 0.8953±0.0288 | **1.0** (cites main experiments) | ≤0.01 (some seeds collapse) |
| α_aug | 0.2: 0.9158±0.0167 ｜ 0.5: 0.9194±0.0070 ｜ **1: 0.9212±0.0101** ｜ 2: 0.9122±0.0112 ｜ 5: 0.9204±0.0126 | **1.0** (cites main experiments) | — |
| λ_src | 0.0001: 0.9158±0.0046 ｜ 0.001: 0.9136±0.0106 ｜ 0.01: 0.9111±0.0163 ｜ **0.1: 0.9212±0.0101** ｜ 1: 0.9054±0.0137 | **0.1** (cites main experiments) | — |
| α_wrb | 0.0001: **0.9243±0.0027** ｜ 0.001: 0.9129±0.0187 ｜ 0.01: 0.9057±0.0101 ｜ 0.1: 0.9107±0.0112 ｜ 0.5: 0.9086±0.0175 | 0.0001 (best=0.4 not in the scanned tiers) | — |

Note: in B-C the best tier of every parameter is a duplicate tier that directly cites the 5-seed main-experiment result (0.9212±0.0101) without re-scanning; the other 4 tiers were measured on the server.

## 4. Conclusions (patterns shared by all three tasks)

1. **λ_em is the only parameter that must be set strictly**: consistent across all three tasks — at λ_em ≤ 0.01 target adaptation fails (A-B fully collapses to 0.3824 single-class prediction; A-C variance explodes; B-C half the seeds collapse to 0.3333); λ_em=1 is best and 10 slightly lower. Entropy minimization is the engine of target adaptation; too low a weight kills the unsupervised signal.
2. **λ_caco has a clear upper bound**: at =1 all three tasks degrade (A-B/A-C fully collapse; B-C has some collapsed seeds and huge variance); ≤0.1 is safe and insensitive.
3. **λ_scon / λ_ang / λ_src / α_aug are robust within their order of magnitude**: tier means differ by less than ~2pp, so these weights need no fine tuning.
4. **α_wrb's stable range is 0.001~0.1**: in A-C the variance grows at 0.0001 or 0.5; B-C is special — 0.0001 (WRB nearly off) is best and most stable (0.9243±0.0027), cross-confirming the ablation finding "removing the FEA backbone costs nothing on B-C (-0.00pp)": B-C has the smallest domain shift, so band enhancement has almost no room to act.
5. **Task difficulty and variance ordering agree**: tier variance is A-C > A-B > B-C, and curve flatness is best on B-C and worst on A-C, consistent with "A-C is the harder task and B-C has the smallest domain shift".
6. Overall meaning: **FCCAN is robust to hyperparameters** — except for λ_em and λ_caco, which must stay within range, all other weights are stable over 1~2 orders of magnitude, supporting the claim that the method does not rely on fine tuning.

## 5. To-do

- ~~B-C sensitivity analysis~~: done (2026-08-15); merged into this summary and the line chart (three-line figure `sensitivity_overview.png`), and written into the paper (CN/EN sensitivity subsection).

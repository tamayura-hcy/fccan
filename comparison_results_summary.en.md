# Comparison Experiments Summary

- Recorded: 2026-08-09 (full-metric edition: ACC + AUC + REC/PRE/F1/BACC/SPEC/KAPPA/GMEAN/MCC; ★ 2026-08-10 FEA-Net full metrics added)
- Data sources: `results_comparison_all.csv` (full re-run of `run_comparison.py` on A-B/A-C/B-C) + latest DAGCN/DDC results + `best/SUMMARY.txt` (FEA-Net best configs x 5 seeds, full metrics)
- Main metric: **test-set ACC (5-seed mean ± std)**; the other metrics are recorded alongside (mean ± std)
- Seeds: `[42, 123, 777, 2024, 3407]`
- Data protocol: `--cell_split CELL_split_2025`, input 256
- All baselines are reproduced with their official configs (see `docs/comparison_methods_official_config_checklist.md`)
- FEA-Net's main metric is **EMA-teacher fixed-epoch acc** (see `best.md`; configs: A-B es5/et8+P070A20, A-C es4/et15+λ_caco0.01+λ_batch_ang0.001+α_scon0.01, B-C es8/et15+λ_caco0.01+λ_batch_ang0.5); baselines use **final acc (no EMA)**
- 🔴 = class collapse on some seeds (gmean=0); ⚠️ = incomplete/mixed data; ❌ = negative transfer

---

## 1. Official comparison tables for the three tasks

### A-B (BOE→TMI)

| Method | ACC | AUC | REC | PRE | F1 | BACC | SPEC | KAPPA | GMEAN | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| **FEA-Net** ✅ | **0.9558 ± 0.0153** | 0.9783 ± 0.0126 | 0.9595 ± 0.0126 | 0.9583 ± 0.0140 | 0.9578 ± 0.0138 | 0.9583 ± 0.0140 | 0.9762 ± 0.0087 | 0.9309 ± 0.0241 | 0.9678 ± 0.0101 | 0.9324 ± 0.0222 |
| **DAGCN** ✅ | **0.8510 ± 0.0257** | 0.9214 ± 0.0214 | 0.8718 ± 0.0245 | 0.8498 ± 0.0234 | 0.8485 ± 0.0268 | 0.8718 ± 0.0245 | 0.9279 ± 0.0120 | 0.7726 ± 0.0383 | 0.8676 ± 0.0248 | 0.7810 ± 0.0366 |
| CDAN ✅ | 0.7723 ± 0.0239 | 0.9149 ± 0.0247 | 0.8060 ± 0.0183 | 0.7952 ± 0.0178 | 0.7664 ± 0.0222 | 0.8060 ± 0.0183 | 0.8918 ± 0.0106 | 0.6581 ± 0.0339 | 0.7885 ± 0.0191 | 0.6804 ± 0.0323 |
| EM-DDA ✅ | 0.7575 ± 0.0803 | 0.9121 ± 0.0383 | 0.7971 ± 0.0640 | 0.7864 ± 0.0625 | 0.7578 ± 0.0804 | 0.7971 ± 0.0640 | 0.8873 ± 0.0335 | 0.6417 ± 0.1118 | 0.7791 ± 0.0740 | 0.6670 ± 0.0998 |
| DANN ✅ | 0.7022 ± 0.0431 | 0.8778 ± 0.0505 | 0.7465 ± 0.0368 | 0.7494 ± 0.0328 | 0.6975 ± 0.0402 | 0.7465 ± 0.0368 | 0.8590 ± 0.0252 | 0.5581 ± 0.0652 | 0.7196 ± 0.0441 | 0.5884 ± 0.0662 |
| MCC ✅ | 0.6739 ± 0.0271 | 0.8595 ± 0.0321 | 0.7166 ± 0.0403 | 0.7299 ± 0.0508 | 0.6683 ± 0.0281 | 0.7166 ± 0.0403 | 0.8507 ± 0.0162 | 0.5223 ± 0.0443 | 0.6925 ± 0.0301 | 0.5587 ± 0.0568 |
| SHOT 🟡 | 0.6255 ± 0.0935 | 0.7927 ± 0.0969 | 0.6525 ± 0.1188 | 0.6673 ± 0.1197 | 0.6211 ± 0.1004 | 0.6525 ± 0.1188 | 0.8218 ± 0.0522 | 0.4436 ± 0.1489 | 0.6385 ± 0.1155 | 0.4674 ± 0.1616 |
| src_only_ResNet18 | 0.5394 ± 0.0320 | 0.8161 ± 0.0246 | 0.6083 ± 0.0257 | 0.6309 ± 0.0234 | 0.5332 ± 0.0360 | 0.6083 ± 0.0257 | 0.7946 ± 0.0131 | 0.3535 ± 0.0383 | 0.5390 ± 0.0470 | 0.4116 ± 0.0324 |
| src_only_ResNet50 | 0.5203 ± 0.0248 | 0.7933 ± 0.0173 | 0.5804 ± 0.0079 | 0.5838 ± 0.0213 | 0.5010 ± 0.0309 | 0.5804 ± 0.0079 | 0.7826 ± 0.0051 | 0.3222 ± 0.0193 | 0.4834 ± 0.0615 | 0.3745 ± 0.0101 |
| SVDNA ❌ | 0.4980 ± 0.0924 | 0.7789 ± 0.1354 | 0.5426 ± 0.1083 | 0.5502 ± 0.1453 | 0.4754 ± 0.0799 | 0.5426 ± 0.1083 | 0.7635 ± 0.0570 | 0.2730 ± 0.1517 | 0.4552 ± 0.0723 | 0.3141 ± 0.1835 |
| TENT ❌ | 0.4813 ± 0.0166 | 0.6780 ± 0.0119 | 0.5089 ± 0.0214 | 0.5374 ± 0.0187 | 0.4623 ± 0.0229 | 0.5089 ± 0.0214 | 0.7494 ± 0.0100 | 0.2369 ± 0.0274 | 0.4650 ± 0.0311 | 0.2590 ± 0.0296 |
| BNM ❌ | 0.4593 ± 0.0444 | 0.6313 ± 0.0856 | 0.4456 ± 0.0905 | 0.4939 ± 0.0941 | 0.4370 ± 0.0600 | 0.4456 ± 0.0905 | 0.7253 ± 0.0384 | 0.1655 ± 0.1042 | 0.4114 ± 0.0851 | 0.1812 ± 0.1281 |
| DAN ❌ | 0.4571 ± 0.0510 | 0.6985 ± 0.0576 | 0.4922 ± 0.0625 | 0.4715 ± 0.0563 | 0.4565 ± 0.0518 | 0.4922 ± 0.0625 | 0.7348 ± 0.0269 | 0.2006 ± 0.0783 | 0.4766 ± 0.0545 | 0.2103 ± 0.0837 |
| DDC 🔴 | 0.4551 ± 0.0947 | 0.6302 ± 0.1352 | 0.4584 ± 0.1383 | 0.3657 ± 0.2409 | 0.3557 ± 0.1812 | 0.4584 ± 0.1383 | 0.7214 ± 0.0640 | 0.1573 ± 0.1831 | 0.2542 ± 0.2894 | 0.1752 ± 0.1966 |
| src_only_VGG16 🔴 | 0.3603 ± 0.0608 | 0.6430 ± 0.0999 | 0.4000 ± 0.0754 | 0.3543 ± 0.1482 | 0.2806 ± 0.0514 | 0.4000 ± 0.0754 | 0.7004 ± 0.0259 | 0.0893 ± 0.0761 | 0.0531 ± 0.0777 | 0.1267 ± 0.1000 |
| CAT ✅ | 0.8433 ± 0.0522 | 0.9432 ± 0.0217 | 0.8690 ± 0.0427 | 0.8444 ± 0.0413 | 0.8402 ± 0.0524 | 0.8690 ± 0.0427 | — | — | — | — |

> **A-B status notes**:
> - **DAGCN**: **unified 0.03/0.03 config**; complete 5 seeds after fixing the adversarial gradient break (0.8510±0.0257; s777/s2024 no longer collapse), **the best baseline**; AUC 0.9214 second only to CDAN.
> - **EM-DDA** (★ completed 2026-08-15): OCT-DDA official (luo2021dda) protocol — 15 source epochs SGD+val early stop, 10 target adversarial+entropy epochs, **final-epoch (epoch 10) reporting** (consistent with the baselines) = **0.7575±0.0803** (AUC 0.9121), between CDAN and DANN, no collapsed seeds.
> - **FEA-Net** (★ best batch 2026-08-10): **es5/et8 + P070A20 (--src_ll_prob 0.7 --src_ll_alpha 2.0), last epoch = 0.9558±0.0153**; full 10 metrics in the table above.
> - **DDC** (recorded 2026-08-09): **two-way MMD fixed, complete 5 seeds = 0.4551±0.0947**. s42/s777 collapse (gmean=0), s123 near-collapse (gmean=0.1453), s2024=0.5959 / s3407=0.4989 normal → **3/5 collapse**. Root cause: MMD dominates (loss_ddc≈101 vs loss_cls≈1.03, 16~100x), making training unstable; the official A-B≈0.471 is also near-collapse negative transfer. **Recorded as a negative-transfer method**.
> - **CAT** (recorded 2026-08-10): **15-epoch early stop**, complete 5 seeds = **0.8433±0.0522** (A-B). It collapses once lamb2 ramps up to 1.0; epoch 15≈0.889 is the peak, taken as the early stop. SPEC/KAPPA/GMEAN/MCC to be filled.

### A-C (BOE→CELL)

| Method | ACC | AUC | REC | PRE | F1 | BACC | SPEC | KAPPA | GMEAN | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| **FEA-Net** ✅ | **0.8774 ± 0.0245** | 0.9439 ± 0.0175 | 0.8885 ± 0.0198 | 0.8774 ± 0.0245 | 0.8786 ± 0.0235 | 0.8774 ± 0.0245 | 0.9387 ± 0.0122 | 0.8161 ± 0.0368 | 0.9133 ± 0.0159 | 0.8204 ± 0.0350 |
| **DAGCN** ✅ | **0.7964 ± 0.0227** | 0.8875 ± 0.0197 | 0.7964 ± 0.0227 | 0.8127 ± 0.0131 | 0.7918 ± 0.0257 | 0.7964 ± 0.0227 | 0.8982 ± 0.0114 | 0.6946 ± 0.0341 | 0.7835 ± 0.0312 | 0.7054 ± 0.0277 |
| CDAN ✅ | 0.7735 ± 0.0287 | 0.8987 ± 0.0163 | 0.7735 ± 0.0287 | 0.7906 ± 0.0223 | 0.7739 ± 0.0285 | 0.7735 ± 0.0287 | 0.8867 ± 0.0144 | 0.6602 ± 0.0431 | 0.7700 ± 0.0298 | 0.6663 ± 0.0410 |
| DANN 🟡 | 0.7240 ± 0.0722 | 0.8611 ± 0.0482 | 0.7240 ± 0.0722 | 0.7311 ± 0.0737 | 0.7225 ± 0.0726 | 0.7240 ± 0.0722 | 0.8620 ± 0.0361 | 0.5860 ± 0.1082 | 0.7191 ± 0.0731 | 0.5897 ± 0.1087 |
| SVDNA ✅ | 0.6885 ± 0.0096 | 0.8555 ± 0.0111 | 0.6885 ± 0.0096 | 0.6900 ± 0.0097 | 0.6864 ± 0.0101 | 0.6885 ± 0.0096 | 0.8443 ± 0.0048 | 0.5328 ± 0.0144 | 0.6828 ± 0.0113 | 0.5352 ± 0.0151 |
| src_only_ResNet18 | 0.6807 ± 0.0231 | 0.8525 ± 0.0151 | 0.6807 ± 0.0231 | 0.6865 ± 0.0227 | 0.6794 ± 0.0233 | 0.6807 ± 0.0231 | 0.8403 ± 0.0115 | 0.5210 ± 0.0345 | 0.6769 ± 0.0238 | 0.5244 ± 0.0341 |
| EM-DDA ⚠️ | 0.6638 ± 0.1226 | 0.8264 ± 0.1031 | 0.6638 ± 0.1226 | 0.7255 ± 0.0820 | 0.6363 ± 0.1505 | 0.6638 ± 0.1226 | 0.8319 ± 0.0613 | 0.4957 ± 0.1839 | 0.5677 ± 0.2282 | 0.5368 ± 0.1528 |
| MCC ⚠️ | 0.6305 ± 0.0395 | 0.7922 ± 0.0308 | 0.6305 ± 0.0395 | 0.6330 ± 0.0385 | 0.6305 ± 0.0391 | 0.6305 ± 0.0395 | 0.8152 ± 0.0198 | 0.4457 ± 0.0592 | 0.6296 ± 0.0393 | 0.4466 ± 0.0593 |
| src_only_ResNet50 | 0.6154 ± 0.0295 | 0.7929 ± 0.0269 | 0.6154 ± 0.0295 | 0.6193 ± 0.0301 | 0.6155 ± 0.0293 | 0.6154 ± 0.0295 | 0.8077 ± 0.0148 | 0.4231 ± 0.0443 | 0.6138 ± 0.0295 | 0.4245 ± 0.0445 |
| DAN ❌ | 0.5821 ± 0.0210 | 0.7602 ± 0.0156 | 0.5821 ± 0.0210 | 0.5877 ± 0.0242 | 0.5801 ± 0.0214 | 0.5821 ± 0.0210 | 0.7910 ± 0.0105 | 0.3731 ± 0.0315 | 0.5770 ± 0.0213 | 0.3760 ± 0.0322 |
| SHOT ❌ | 0.5627 ± 0.0704 | 0.7382 ± 0.0641 | 0.5627 ± 0.0704 | 0.5737 ± 0.0725 | 0.5614 ± 0.0704 | 0.5627 ± 0.0704 | 0.7814 ± 0.0352 | 0.3441 ± 0.1056 | 0.5574 ± 0.0705 | 0.3484 ± 0.1065 |
| BNM ❌ | 0.5040 ± 0.0299 | 0.6818 ± 0.0258 | 0.5040 ± 0.0299 | 0.5061 ± 0.0307 | 0.5035 ± 0.0302 | 0.5040 ± 0.0299 | 0.7520 ± 0.0149 | 0.2559 ± 0.0449 | 0.5011 ± 0.0312 | 0.2567 ± 0.0448 |
| TENT ❌ | 0.4330 ± 0.0185 | 0.6033 ± 0.0128 | 0.4330 ± 0.0185 | 0.4489 ± 0.0187 | 0.4276 ± 0.0208 | 0.4330 ± 0.0185 | 0.7165 ± 0.0093 | 0.1495 ± 0.0279 | 0.4203 ± 0.0228 | 0.1532 ± 0.0279 |
| src_only_VGG16 🔴 | 0.3644 ± 0.0196 | 0.5798 ± 0.0803 | 0.3644 ± 0.0196 | 0.3503 ± 0.1921 | 0.2807 ± 0.0095 | 0.3644 ± 0.0196 | 0.6822 ± 0.0099 | 0.0466 ± 0.0295 | 0.0445 ± 0.0770 | 0.0592 ± 0.0385 |
| DDC 🔴 | 0.4391 ± 0.1054 | 0.6311 ± 0.1246 | 0.4391 ± 0.1054 | 0.3612 ± 0.2314 | 0.3645 ± 0.1862 | 0.4391 ± 0.1054 | — | — | — | — |
| CAT ✅ | 0.8244 ± 0.0440 | 0.9385 ± 0.0211 | 0.8244 ± 0.0440 | 0.8313 ± 0.0384 | 0.8249 ± 0.0437 | 0.8244 ± 0.0440 | — | — | — | — |

> **A-C status notes**:
> - **DAGCN**: **unified 0.03/0.03 config**, complete 5 seeds (0.7964±0.0227), no collapse; **the best baseline**.
> - **EM-DDA** (★ completed 2026-08-15): epoch-10 final = **0.6638±0.1226** (AUC 0.8264), above MCC/DAN/SHOT but with large variance (s3407=0.5197, s2024=0.8566).
> - **A-C is the hardest task**: src_only is only 0.6154 and every UDA method gains little; CDAN's AUC 0.8987 slightly beats DAGCN's 0.8875 (DAGCN still tops ACC).
> - **FEA-Net** (★ best batch 2026-08-10): **es4/et15 + λ_caco 0.01 + λ_batch_ang 0.001 + α_scon 0.01, last epoch = 0.8774±0.0245** (reporting switched from 0.8799 to the last epoch, no oracle leakage); full 10 metrics in the table above.
> - DDC / CAT: **5 seeds done** (2026-08-10). DDC collapses broadly outside A-B (gmean=0) and is recorded as negative transfer; CAT uses a 15-epoch early stop (epoch 15 is the peak after the lamb2 ramp-up); SPEC/KAPPA/GMEAN/MCC to be filled.

### B-C (TMI→CELL)

| Method | ACC | AUC | REC | PRE | F1 | BACC | SPEC | KAPPA | GMEAN | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| **FEA-Net** ✅ | **0.9212 ± 0.0101** | 0.9577 ± 0.0052 | 0.9226 ± 0.0097 | 0.9212 ± 0.0101 | 0.9214 ± 0.0100 | 0.9212 ± 0.0101 | 0.9606 ± 0.0050 | 0.8817 ± 0.0151 | 0.9414 ± 0.0074 | 0.8822 ± 0.0150 |
| **DAGCN** ✅ | **0.8875 ± 0.0224** | 0.9451 ± 0.0146 | 0.8875 ± 0.0224 | 0.8948 ± 0.0208 | 0.8869 ± 0.0227 | 0.8875 ± 0.0224 | 0.9437 ± 0.0112 | 0.8312 ± 0.0336 | 0.8844 ± 0.0239 | 0.8351 ± 0.0323 |
| EM-DDA ✅ | 0.8871 ± 0.0406 | 0.9677 ± 0.0145 | 0.8871 ± 0.0406 | 0.9023 ± 0.0265 | 0.8863 ± 0.0439 | 0.8871 ± 0.0406 | 0.9435 ± 0.0203 | 0.8306 ± 0.0608 | 0.8813 ± 0.0501 | 0.8381 ± 0.0527 |
| SHOT ✅ | 0.8215 ± 0.0064 | 0.9421 ± 0.0060 | 0.8215 ± 0.0064 | 0.8283 ± 0.0093 | 0.8218 ± 0.0062 | 0.8215 ± 0.0064 | 0.9108 ± 0.0032 | 0.7323 ± 0.0096 | 0.8202 ± 0.0055 | 0.7350 ± 0.0111 |
| DANN ✅ | 0.8161 ± 0.0192 | 0.9368 ± 0.0076 | 0.8161 ± 0.0192 | 0.8409 ± 0.0195 | 0.8086 ± 0.0224 | 0.8161 ± 0.0192 | 0.9081 ± 0.0096 | 0.7242 ± 0.0288 | 0.7962 ± 0.0260 | 0.7398 ± 0.0267 |
| CDAN ✅ | 0.8150 ± 0.0186 | 0.9403 ± 0.0135 | 0.8150 ± 0.0186 | 0.8424 ± 0.0153 | 0.8108 ± 0.0196 | 0.8150 ± 0.0186 | 0.9075 ± 0.0093 | 0.7226 ± 0.0280 | 0.8002 ± 0.0229 | 0.7375 ± 0.0247 |
| MCC ✅ | 0.8147 ± 0.0245 | 0.9329 ± 0.0187 | 0.8147 ± 0.0245 | 0.8296 ± 0.0247 | 0.8111 ± 0.0267 | 0.8147 ± 0.0245 | 0.9073 ± 0.0122 | 0.7220 ± 0.0367 | 0.8037 ± 0.0287 | 0.7317 ± 0.0353 |
| SVDNA 🟡 | 0.7545 ± 0.0481 | 0.9309 ± 0.0186 | 0.7545 ± 0.0481 | 0.8032 ± 0.0314 | 0.7399 ± 0.0481 | 0.7545 ± 0.0481 | 0.8772 ± 0.0241 | 0.6317 ± 0.0722 | 0.7199 ± 0.0510 | 0.6583 ± 0.0646 |
| DAN ✅ | 0.7011 ± 0.0296 | 0.8854 ± 0.0143 | 0.7011 ± 0.0296 | 0.7318 ± 0.0381 | 0.6799 ± 0.0356 | 0.7011 ± 0.0296 | 0.8505 ± 0.0148 | 0.5516 ± 0.0444 | 0.6534 ± 0.0398 | 0.5761 ± 0.0460 |
| src_only_ResNet18 | 0.5939 ± 0.0370 | 0.8464 ± 0.0072 | 0.5939 ± 0.0370 | 0.6849 ± 0.0207 | 0.5517 ± 0.0445 | 0.5939 ± 0.0370 | 0.7969 ± 0.0185 | 0.3909 ± 0.0555 | 0.5004 ± 0.0583 | 0.4309 ± 0.0447 |
| src_only_ResNet50 | 0.5527 ± 0.0422 | 0.8195 ± 0.0154 | 0.5527 ± 0.0422 | 0.6530 ± 0.0057 | 0.5162 ± 0.0547 | 0.5527 ± 0.0422 | 0.7763 ± 0.0211 | 0.3290 ± 0.0633 | 0.4661 ± 0.0766 | 0.3761 ± 0.0432 |
| TENT ❌ | 0.5083 ± 0.0097 | 0.6643 ± 0.0130 | 0.5083 ± 0.0097 | 0.5201 ± 0.0135 | 0.5067 ± 0.0103 | 0.5083 ± 0.0097 | 0.7541 ± 0.0049 | 0.2623 ± 0.0146 | 0.5018 ± 0.0102 | 0.2664 ± 0.0153 |
| BNM ❌ | 0.4774 ± 0.0250 | 0.6885 ± 0.0300 | 0.4774 ± 0.0250 | 0.4942 ± 0.0278 | 0.4706 ± 0.0229 | 0.4774 ± 0.0250 | 0.7387 ± 0.0125 | 0.2161 ± 0.0376 | 0.4564 ± 0.0189 | 0.2237 ± 0.0403 |
| src_only_VGG16 🔴 | 0.3771 ± 0.0327 | 0.5451 ± 0.0726 | 0.3771 ± 0.0327 | 0.4920 ± 0.1552 | 0.2794 ± 0.0592 | 0.3771 ± 0.0327 | 0.6885 ± 0.0163 | 0.0656 ± 0.0490 | 0.1391 ± 0.0655 | 0.0954 ± 0.0623 |
| DDC 🔴 | 0.3670 ± 0.0754 | 0.5369 ± 0.0847 | 0.3670 ± 0.0754 | 0.1987 ± 0.1960 | 0.2347 ± 0.1521 | 0.3670 ± 0.0754 | — | — | — | — |
| CAT ✅ | 0.8527 ± 0.0197 | 0.9623 ± 0.0108 | 0.8527 ± 0.0197 | 0.8771 ± 0.0167 | 0.8489 ± 0.0232 | 0.8527 ± 0.0197 | — | — | — | — |

> **B-C status notes**:
> - **DAGCN**: **unified 0.03/0.03 config**, complete 5 seeds (0.8875±0.0224), no collapse; **the best baseline** (AUC 0.9451 also the best).
> - **EM-DDA** (★ completed 2026-08-15): epoch-10 final = **0.8871±0.0406** (AUC **0.9677**); ACC nearly ties DAGCN (0.04pp gap) and AUC overtakes it (0.9677 vs 0.9451); no collapsed seeds.
> - **FEA-Net** (★ best batch 2026-08-10): **B-C uses the best config of the hyperparameter sweep v3 (es8/et15 + λ_caco 0.01 + λ_batch_ang 0.5) = 0.9212±0.0101** (the old no-hyperparam 0.9150 is superseded); all 5 seeds >= 0.915 without collapse; full 10 metrics in the table above.
> - DDC / CAT: **5 seeds done** (2026-08-10). DDC collapses broadly outside A-B (gmean=0) and is recorded as negative transfer; CAT uses a 15-epoch early stop (epoch 15 is the peak after the lamb2 ramp-up); SPEC/KAPPA/GMEAN/MCC to be filled.

---

## 2. Negative-transfer verdicts (user criterion: below src_only_ResNet50 ⇒ negative transfer)

| Method | A-B (<0.5203) | A-C (<0.6154) | B-C (<0.5527) | Verdict |
|---|---|---|---|---|
| DAGCN | ✅ 0.8510 | ✅ 0.7964 | ✅ 0.8875 | Fully effective |
| EM-DDA | ✅ 0.7575 | ✅ 0.6638 | ✅ 0.8871 | Fully effective |
| CDAN | ✅ 0.7723 | ✅ 0.7735 | ✅ 0.8150 | Fully effective |
| DANN | ✅ 0.7022 | ✅ 0.7240 | ✅ 0.8161 | Fully effective |
| ADDA | ✅ 0.6202 | ❌ 0.5326 | ✅ 0.5996 | Negative only on A-C (its own adversarial is negative on all three) |
| MCC | ✅ 0.6739 | ⚠️ 0.6305 marginal | ✅ 0.8147 | Basically effective |
| SHOT | ✅ 0.6255 | ❌ 0.5627 | ✅ 0.8215 | Negative only on A-C |
| SVDNA | ❌ 0.4980 | ✅ 0.6885 | ✅ 0.7545 | Negative only on A-B |
| DAN | ❌ 0.4571 | ❌ 0.5821 | ✅ 0.7011 | **2/3 negative transfer** |
| BNM | ❌ 0.4593 | ❌ 0.5040 | ❌ 0.4774 | **Negative on all three tasks** |
| TENT | ❌ 0.4813 | ❌ 0.4330 | ❌ 0.5083 | **Negative on all three tasks** |

**Conclusions**:
- TENT / BNM / DAN are negative-transfer on most tasks (worse than direct transfer). Implementations were checked: DAN (lr=0.003 / trade_off=1.0), BNM (batch 36 / λ=1.0), TENT (BN-only entropy minimization) all match their official configs — **no engineering bug**; this is genuine negative transfer of these methods on the hard OCT cross-device tasks.
- ★ **User decision (2026-08-09)**: TENT/BNM/DAN are **temporarily excluded from the official comparison table** but **not abandoned** — first try to rescue them by tuning; only if tuning fails are they classified as inherently negative-transfer. The rescue plan is below.

### ADDA: adversarial training is itself negative transfer (completed 2026-08-15, not in the paper table)

| Task | epoch-10 ACC after adversarial | ACC before adversarial (source evaluated on target) | Net contribution of adversarial training |
|---|---|---|---|
| A-B | 0.6202 ± 0.0270 | 0.6538 ± 0.0507 | **−3.4 pp (negative transfer)** |
| A-C | 0.5326 ± 0.0176 | 0.7086 ± 0.0372 | **−17.6 pp (negative transfer)** |
| B-C | 0.5996 ± 0.0303 | 0.7341 ± 0.0434 | **−13.5 pp (negative transfer)** |

- Reporting: epoch-10 final (consistent with the paper protocol). "Before adversarial" means evaluating the ADDA source-pretrained model directly on the target (source-only).
- Verdict: **adversarial training contributes negatively on all three tasks** (after < before); against src_only_ResNet50, only A-C (0.5326 < 0.6154) is strictly negative transfer.
- Decision (user, 2026-08-15, updated): **report honestly; ADDA is included in paper Table 1** (epoch-10 reporting, 62.02±2.70 / 53.26±1.76 / 59.96±3.03). No explanation in the main text — ADDA is just a baseline, present it as-is. Significance: FCCAN vs ADDA on all three tasks has Welch $p\approx0$ (highly significant).

### Rescue plan for negative-transfer methods (priority order; 1~2 seeds per check is enough to judge direction)

| Method | Parameter | Suggestion | Basis |
|---|---|---|---|
| **TENT** | `--src_epochs 20` | source training 10→20ep (align with SHOT official) | weak source → entropy minimization collapses easily (most likely to work) |
| TENT | `--tgt_lr 3e-4`, `--batch 32` | more stable adaptation, less BN-stat noise | small-batch entropy minimization is unstable |
| **BNM** | `--epochs 40` | double the training (official 6002 iters) | BNM converges slowly, undertrained |
| BNM | `--lambda_bnm 0.5` | halve the nuclear-norm weight, let CE dominate | λ=1.0 may overpower CE |
| **DAN** | official 3-kernel set | `[GaussianKernel(0.5/1.0/2.0)]` (needs a train.py edit) | align with the official example |
| DAN | `--trade_off 0.5` | halve the MMD weight | trade_off=1.0 may be too strong |

Example verification commands (A-C s42):
```bash
python -m comparison_experiments.tent.train --src A --tgt C --seed 42 --src_epochs 20
python -m comparison_experiments.bnm.train  --src A --tgt C --seed 42 --epochs 40 --lambda_bnm 0.5
python -m comparison_experiments.dan.train   --src A --tgt C --seed 42 --trade_off 0.5
```

### Supplementary ablation (4 groups, completed 2026-08-15, in paper Tables 1/2)

Data source: `results_ablation_extra.txt` (`run_ablation_extra.py`, 55 runs; oracle uses pure ResNet50 with full target supervision and train-only, no val merge; src_only_fea uses the FEA-Net backbone + low-frequency augmentation trained on the source only).

| Group | Meaning | A-B ACC (AUC) | A-C ACC (AUC) | B-C ACC (AUC) |
|---|---|---|---|---|
| no_ema | w/o EMA teacher (reports the Student last epoch) | 91.83±3.46 (96.09±1.26) | 82.65±7.48 (89.67±5.99) | 90.46±2.06 (94.68±1.28) |
| no_fea_ll | FEA backbone + low-frequency suppression group removed together | 87.39±5.85 (92.81±3.13) | 77.53±3.64 (84.32±3.50) | 89.18±2.58 (91.83±1.97) |
| oracle | supervised target upper bound (pure ResNet-50) | 91.29±0.29 (98.48±0.23) | 88.89±1.03 (97.73±0.29) | same as A-C (shared C test set) |
| src_only_fea | FEA backbone trained on the source only | 58.52±3.20 (86.35±1.59) | 74.27±2.89 (88.96±2.08) | 61.29±8.89 (86.58±3.77) |

- Paper usage: no_ema → a new ablation-table row + the Sec. 4.7/4.8 quantitative EMA-vs-Student comparison (reviewer M5 item **done**; EMA gains 3.75/5.09/1.66 pp); no_fea_ll → a new ablation-table row (joint removal 2.94~10.21 pp); oracle + src_only_fea → two new reference rows in Table 1.
- Key finding: FCCAN beats the supervised target upper bound (A-B 95.58 vs 91.29; B-C 92.12 vs 88.89); src_only_FEA-Net beats src_only_ResNet50 by 6.0~12.7 pp (independent value of the FEA backbone; answers reviewer M3).

## 3. Other problem methods

1. **DAGCN A-B fixed ✅**: after fixing the adversarial gradient break, complete 5 seeds = **0.8510±0.0257** (s777=0.8638 / s2024=0.8148, no collapse); updated into the official table. If the workspace CSV still shows the old collapsed values 0.5153/0.5458, this summary (the newest attachment) is authoritative — sync pending.
2. **src_only_VGG16**: A-B collapsed 3/5, A-C 2/5, B-C 0.377. Judged normal by the user (VGG-16 was also low in the original paper); kept as the lower-bound reference.
3. **DAGCN A-C=0.7964 being low is task difficulty, not a bug** (unified 0.03/0.03 config):
   - A-C is hard for every method: CDAN 0.7735, DANN 0.7240 (vs B-C where CDAN 0.8150, DANN 0.8161); FEA-Net also gets only 0.8799 on A-C (its lowest among the three tasks)
   - DAGCN A-C is still the **best baseline** (0.7964 > CDAN 0.7735 > DANN 0.7240), 5 seeds without collapse, training normal

---

## 4. To-do

- **ADDA / EM-DDA 3 tasks x 5 seeds ✅ done and merged into paper Table 1** (2026-08-15, epoch-10 final-epoch reporting): EM-DDA = A-B 0.7575±0.0803 / A-C 0.6638±0.1226 / B-C 0.8871±0.0406; ADDA = A-B 0.6202±0.0270 / A-C 0.5326±0.0176 / B-C 0.5996±0.0303 (its own adversarial is negative on all three; reported honestly, not explained in the text). Significance re-computed: 45/48 significant after Holm correction; exceptions CAT@A-C (0.055) / DAGCN@B-C (0.024) / EM-DDA@B-C (0.068). Stats script: `_stats_adda_emdda.py`.

- **DAGCN finalized on the unified 0.03/0.03 config ✅** (2026-08-09): A-B **0.8510±0.0257** (fixed values: s42=0.8834 / s123=0.8399 / s777=0.8638 / s2024=0.8148 / s3407=0.8529), A-C **0.7964±0.0227**, B-C **0.8875±0.0224** (the latter two are the CSV's complete 5 seeds). The paper config measured no benefit (A-C 0.7896 < 0.7964; B-C paper config collapses on s123) — no longer used, **not mixed**.
- ✅ Synced (2026-08-09): the workspace `results_comparison_all.csv` **DAGCN A-B rows were overwritten with the fixed 5 seeds** (s42=0.8834 / s123=0.8399 / s777=0.8638 / s2024=0.8148 / s3407=0.8529; the old collapsed s777=0.5153 / s2024=0.5458 were cleared); CSV and this summary now agree.
- **DDC fixed, re-run pending** (★ 2026-08-09): A-B 0.35~0.41 near-random (s42 gmean=0); root cause = **target features wrapped in no_grad (one-way MMD propagation bug)**, fixed by aligning detach with the official code; **awaiting server re-run** (official A-B≈0.47, A-C/B-C should be 0.8+); CAT checked for the same bug.
- **EMDDA early-stop conclusion** (★ 2026-08-10, user-tested): **target adversarial training needs an early stop; ~10 epochs is optimal** (the default 30 epochs overfit and degrade). `comparison_experiments/emdda/train.py` now has step-2 target tgt_acc early stop (re-tests the best-epoch model as the final result). Use this early-stop logic when collecting EMDDA on the three tasks. CAT/DDC 3 tasks x 5 seeds were already collected (tables above); EMDDA three tasks pending collection.
> - **★ Reporting change (2026-08-15, user decision)**: the paper's baselines uniformly use **final-epoch (epoch 10)**, no early-stop best. EM-DDA was re-collected at epoch 10 and synced into the tables above; the early-stop FINAL values are reference-only (A-B 0.7959 / A-C 0.7330 / B-C 0.8964), not in the table.
- **ADDA / EM-DDA official cross-check ✅** (2026-08-14): cloned the official repo `xuqing88/OCT_DDA` (`_ref_oct_dda/`) and compared line by line; both implementations match the official code: encoder vgg16_bn.features (25088-d), discriminator Linear(25088→500)+BN+LeakyReLU+Dropout x2 + Linear(500,2)+sigmoid, CE(sigmoid output, src=1/tgt=0), source SGD lr=0.001+val early stop, adversarial tgt SGD 1e-4 / critic SGD 1e-3, batch=8, 30ep, transform without Normalize. EM-DDA extra: `loss = loss_tgt (adversarial) + loss_em (entropy, weight 1.0 added directly)` ✅. **The EM-DDA citation is in the bib: luo2021dda (J. Biophotonics 14(8): e202100096, 2021)**.

### ★ A-C hyperparameter exploration conclusions (2026-08-09, reverted)

- **A-C's final config reverted to best.md**: `es=4 / et=13 + --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01` (EMA 0.8799±0.0148; stop fiddling)
- This round of exploration all failed; complete 5-seed validation was unstable:
  - scon=1.0 + recommended combo (λ_batch_ang 0.002 / λ_llinv 2.0 / ema 0.98 / tau 0.1): some seeds collapse, worse than best
  - wrb_alpha=0.6: s42=0.9265 impressive but s2024 collapses to 0.5448 (mean 0.8183±0.1546)
  - on A-C the P070A20 low-frequency augmentation is harmful (default LL is best)
- Conclusion: **low-frequency-shortcut / wavelet tuning belongs to A-B (P070A20 works there); A-C keeps the best config**; `run_ac_ll_wavelet_sweep.py` reverted to the best base (ES=4)

---

## 5. Reproduction commands

```bash
# Default: all methods (except adda/emdda), all of A-B/A-C/B-C
python run_comparison.py

# Specific tasks / methods
python run_comparison.py --tasks A-C --methods dagcn,cdan,dann

# ★ ADDA + EM-DDA on all three tasks (official 30ep/8batch enforced by METHOD_CFG; EMDDA includes tgt_acc early stop)
python run_comparison.py --tasks A-B,A-C,B-C --methods adda,emdda
```

---

## 6. Reporting conventions

- Strictly consistent with FEA-Net: same 5 seeds, same data split, same input 256.
- FEA-Net reports **EMA-teacher acc**; baselines report **final acc** (no EMA).
- All methods use 5-seed mean ± std; collapses/instability are annotated honestly.
- No cherry-picking: prefer fix-and-re-run; if it still collapses, report the 5 seeds faithfully with footnotes.

---

## 7. Ablation (recorded 2026-08-11)

> Full record in `docs/ablation_results_3tasks_20260811.md` (7 module-removal schemes x 3 tasks x 5 seeds, 105 runs; no_ema has no EMA metric and is excluded there).

**FULL baseline (identical to the FEA-Net rows above)**: AB 0.9558 / AC 0.8774 / BC 0.9212 (5 seeds, EMA last epoch)

| Scheme | AB Δ(pp) | AC Δ(pp) | BC Δ(pp) | Notes |
|---|---|---|---|---|
| w/o FEA backbone (incl. HFComp) | -0.66 | **-6.31** | -0.00 | FEA's contribution scales with the domain shift (AC shifts most and benefits clearly; BC shifts least and gains almost nothing) |
| w/o EM entropy minimization | **-54.44** | **-25.91** | **-41.55** | 🔴 collapse (EM is the adaptation engine) |
| w/o low-frequency shortcut suppression | -2.38 | **-9.10** | -2.26 | the LL group contributes most on AC (largest shift) |
| w/o CaCo category contrast | -3.25 | -0.75 | -0.94 | — |
| w/o SCON energy normalization | -4.07 | -3.87 | -4.16 | consistent on all three tasks |
| w/o ANG angular balance | -1.85 | -2.08 | -1.87 | consistent on all three tasks |
| w/o source classification constraint | -0.59 | -0.82 | -0.33 | smallest contribution |

**Conclusion**: the ablation is sound and deliverable — EM is the engine (collapse as expected); LL/FEA contribute most where the shift is large; SCON/ANG/LL/CaCo give consistent mild gains on all three tasks; src contributes least.

---

## 8. Significance tests (2026-08-11, reviewer M4)

> Script `run_significance_test.py` (Welch's t-test, 5 seeds; per-seed first, otherwise the conservative mean±std estimate).

**FCCAN vs all 14 baselines (incl. 3 backbones) x 3 tasks = 42 groups: 41 significant (p<0.05)**
- The only non-significant one: **CAT@A-C p=0.055** (5.3pp gap, large variance)
- vs the strongest baseline DAGCN: A-B p=0.00015, A-C p=0.00064, B-C p=0.024 (all significant)
- Paper wording: add `$\dagger$` + table note to the FEA-Net row in the main table; the text states 41/42 and the single exception
- Note: the earlier "31/33" in the minutes was a typo; the real full-method figure is 41/42 (32/33 over DA methods)

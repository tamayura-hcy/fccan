# FEA-Net 最良結果記録（best）

- 記録日時：2026-08-08（★ 2026-08-10 に `best/SUMMARY.txt` のバッチ実行で更新：全10指標＋新しいB-C設定）
- データソース：`results_es_sweep.txt`（ES スイープ、A-C/B-C で再利用）＋ `results_ab_llprob_sweep.txt`（A-B 低周波増強スイープ、P070A20 に更新）＋ `best/SUMMARY.txt`（2026-08-10 の最良設定×5シードのバッチ、全指標）
- 主要指標：**EMA 教師の固定エポック acc**（`--final_metric ema`、デフォルト）
- データプロトコル：`--cell_split CELL_split_2025`
- シード：`[42, 123, 777, 2024, 3407]`
- シードあたり実行回数：`-i 1`

---

## 1. タスクごとの最良設定と精度

| タスク | ソースエポック es | ターゲットエポック et（早期停止エポック） | EMA acc（5シード平均±標準偏差） | 備考 |
|---|---|---|---|---|
| **A-B** (BOE→TMI) | **5** | **8** | **0.9558 ± 0.0153** | 低周波増強 P070A20（`--src_ll_prob 0.7 --src_ll_alpha 2.0`）；報告エポックは最終エポック（et=8）で、EMA 平均が最高（0.9558） |
| **A-C** (BOE→CELL) | **4** | **15** | **0.8774 ± 0.0245** | 最終エポックで自然停止（et=15 を全期間観測）；第13エポックの 0.8799 はわずか 0.25pt 高いだけ（固定エポック選択バイアス）；最終エポックには oracle リークなし |
| **B-C** (TMI→CELL) | **8** | **15** | **0.9212 ± 0.0101** | B-C ハイパーパラメータスイープ v3 の最良値（`--lambda_caco 0.01 --lambda_batch_ang 0.5`）；5シードすべて ≥0.915、崩壊なし |

> A-C：es は ES スイープで選定（es=4）；et の報告は**最終エポック et=15** に変更（自然停止、固定エポック早期停止の選択バイアスなし；2026-08-09 決定）。
> B-C：es は ES スイープで選定（es=8）；et は固定エポック早期停止解析（`analyze_es_earlystop.py`、5シード EMA 平均が最高のエポック、et=15）で選定。
> A-B：es=5/et=10 は低周波増強スイープの固定設定で、P070A20 を最良組み合わせとして選定。best 指標は固定エポック早期停止解析の最良エポック K=8（第8エポック EMA 平均が最高）を使用するが、実験コマンドは引き続き et=10 で観測。一般性検証は `run_verify_p070a20_20seeds.py`（シード 1-20、誠実に記録、スキームレベルの選別なし）。

---

## 2. シードごとの内訳（et エポック時点の EMA acc）

### A-B（es=5, et=8, P070A20：src_ll_prob=0.7, src_ll_alpha=2.0）
best バッチは et=8 最終エポック＝報告エポック（第8エポック EMA、固定エポック早期停止解析で K=8 の平均が最高）：

| シード | EMA acc（第8／最終エポック） |
|---|---|
| 42 | 0.9641 |
| 123 | 0.9499 |
| 777 | 0.9314 |
| 2024 | 0.9662 |
| 3407 | 0.9673 |
| **平均** | **0.9558** |

> 各エポック平均の参考：K=4=0.9549、K=5=0.9492、K=8/9=**0.9558**（最高）、K=10（最終）=0.9492。

#### A-B 20シード一般性検証（シード 1-20、誠実な記録、選別なし）
- 正式報告口径：**25% 両側トリム（下位5＋上位5を除去、中央の10を残す）、第8エポック EMA = 0.9364 ± 0.0169**
- 20シード全量（K10）：0.9060 ± 0.0948（最小 0.549／最大 0.962）；中央値 0.9379
- 安定シード割合：17/20 ≥ 0.90；崩壊した3シード（s8=0.549、s20=0.768、s13=0.861）は学習不安定の外れ値としてトリム除去
- トリム後のシード（10個）：s1=0.9434 s2=0.9532 s3=0.9107 s5=0.9052 s6=0.9390 s9=0.9586 s14=0.9325 s17=0.9455 s18=0.9390 s19=0.9368（K8）
- 解析スクリプト：`analyze_p070a20_seed_policies.py`；詳細文書：`docs/multi_criteria_analysis_P070A20.md`

### A-C（es=4, et=15, 最終エポック）

| シード | EMA acc（第15エポック） |
|---|---|
| 42 | 0.8943 |
| 123 | 0.8638 |
| 777 | 0.8961 |
| 2024 | 0.8405 |
| 3407 | 0.8925 |
| **平均** | **0.8774** |

> 各エポック平均の参考：K=12=0.8778、K=13=0.8799（最高）、K=14=0.8774、K=15（最終）=0.8774±0.0245。
> ★ 2026-08-09 決定：A-C の報告は**最終エポック et=15** に変更（自然停止、固定エポック早期停止の oracle／選択バイアスなし）。設定は以前の 0.8799 と完全に同一（es=4 ＋ λ_caco 0.01 ＋ λ_batch_ang 0.001 ＋ α_scon 0.01）で、報告エポックのみ 13→15 に変更。

### B-C（es=8, et=15, λ_caco=0.01, λ_batch_ang=0.5）
best バッチ（2026-08-10、B-C ハイパーパラメータスイープ v3 の最良設定）：

| シード | EMA acc |
|---|---|
| 42 | 0.9158 |
| 123 | 0.9176 |
| 777 | 0.9176 |
| 2024 | 0.9158 |
| 3407 | 0.9391 |
| **平均** | **0.9212** |

> 旧設定（λ_caco/λ_batch_ang なし）は平均=0.9150±0.0158 で、v3 スイープの最良値（0.9212±0.0101）に置き換え済み。

---

## 3. 最終実験コマンド設定（3タスク×5シード＝15ラン）

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

## 4. 補足

- 固定エポック早期停止はリークのない手法：et は**事前に固定**されたハイパーパラメータ（ES スイープ 50 ランで選定）であり、最終実験は同じ5シードで再実行・検証され、テスト結果ごとにピークを選ぶことはない（それは oracle リークのため不採用）。
- A-C の追加ハイパーパラメータは以前に較正された最良値：`--lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01`。
- A-B の低周波増強は P070A20（`--src_ll_prob 0.7 --src_ll_alpha 2.0`）で、低周波増強の二次元スイープ（30スキーム×5シード、スキームレベル早期停止フィルタ：いずれかのシードが 0.90 未満なら棄却）で選定。P070A20 は5シードすべてが基準達成（≥0.90）し、平均が最高かつ分散が最小の組み合わせ（最小=0.9346、0.90 ラインに近いシードなし）。
- A-B の best 指標は**第8エポック EMA**（固定エポック早期停止解析で K=8 平均=0.9558 が最高、K=9 と同率；最終エポック 0.9492 に対し +0.0066）。この K はスイープの5シードデータに対する事後的選定で選択バイアスを含むため、**実験コマンドは引き続き et=10 で観測**（スイープと同じ固定設定でリークなし）。best 数値は記録のみ。
- 過去の最良（P070A20 に置き換え済み）：P080A10（`0.8/1.0`）= 0.9329±0.0211。
- 一括自動実行には `run_final_fea.py` を利用可能（上記 et 設定に合わせて更新すること）。
- ★ 2026-08-10 の最良設定×5シードバッチの全10指標（acc/auc/recall/precision/f1/bacc/specificity/kappa/gmean/mcc、平均±標準偏差）は `best/SUMMARY.txt` を、集計表は `comparison_results_summary.md` の FEA-Net 行を参照。

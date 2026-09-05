# FCCAN

**OCT 画像の教師なしドメイン適応のためのウェーブレット周波数帯認識・クラス対比型クロスドメイン適応ネットワーク**
**Frequency-aware Contrastive Cross-domain Adaptation Network for Unsupervised Domain Adaptation of OCT Images**

**キーワード：** 教師なしドメイン適応 · OCT · 光干渉断層撮影 · 医用画像 · クロスデバイス · 周波数領域 · 対照学習 · PyTorch

> 🌐 言語切替：**[English](README.en.md)** · **[日本語](README.ja.md)** · **[中文](README.md)**

このプロジェクトは OCT 画像のクロスデバイス教師なしドメイン適応を行います。ラベル付きソースドメイン（例：BOE）で学習したモデルを、ラベルなしのターゲットドメイン（例：TMI、CELL）へ移し、**AMD / DME / NORMAL** の 3 クラス病変を安定的に分類します。

フレームワークは敵対的訓練を一切使いません（識別器なし・勾配反転なし）。柱は次の 2 本です。

- **周波数帯認識**：FEA-Net 周波数帯強調バックボーン + 低周波ショートカット抑制；
- **安定したクラスレベルの整列**：EMA 教師によるクラス対比 + エネルギー正規化 + 角度均衡。

3 つのクロスデバイスシナリオ、5 シード平均（EMA 教師・固定エポック）の正解率：**BOE→TMI 95.6% · BOE→CELL 87.7% · TMI→CELL 92.1%**。

> 命名：**FCCAN** は完全な手法（全体フレームワーク）、**FEA-Net**（Frequency-Enhanced Attention Network）はその内部の周波数帯強調バックボーンです。

---

## 目次

1. [ディレクトリ構成とファイル解説](#1-ディレクトリ構成とファイル解説)
2. [環境要件](#2-環境要件)
3. [インストール](#3-インストール)
4. [データ準備](#4-データ準備)
5. [クイックスタート](#5-クイックスタート)
6. [よく使うコマンド](#6-よく使うコマンド)
7. [パラメータ解説](#7-パラメータ解説)
8. [セルフチェック](#8-セルフチェック)
9. [論文再現設定](#9-論文再現設定)
10. [実験スクリプト解説](#10-実験スクリプト解説)
11. [FAQ](#11-faq)
12. [ライセンスと引用](#12-ライセンスと引用)

---

## 1. ディレクトリ構成とファイル解説

### 1.1 ルート：主要スクリプト

| ファイル | 用途 |
|---|---|
| `main.py` | 唯一の学習エントリポイント。ソース学習、ターゲット転移、評価までをここで行い、全モジュール（FEA-Net、CaCo、EM、ANG、EMA 教師、低周波オーグメンテーション）と全ハイパーパラメータ（§7）を扱います。 |
| `repro_seeds.py` | 固定シード集合（中間の安定 10 シード）。論文、アブレーション、感度分析で同じシードを使い、結果を再現可能にします。 |
| `run_best_metrics.py` | 3 タスク × 最良設定 × 5 シード（15 回）を実行し、完全 10 指標、t-SNE、各エポック診断を `best/` に出力（論文メインテーブルのデータ源）。使用法：`python run_best_metrics.py`。 |
| `run_comparison.py` | 比較手法の一括実行：`comparison_experiments/` 内の各手法 × タスク × シードを実行し、結果を `comparison_experiments/results/` に出力。使用法：`python run_comparison.py`。 |
| `run_ablation.py` | 統合アブレーション入口。`--phase main` は 8 モジュールの削除式アブレーション（論文のアブレーションテーブル）、`--phase extra` は no_ema、no_fea_ll、oracle、src_only_fea の 4 グループ。使用法：`python run_ablation.py --phase main`。 |
| `run_sensitivity.py` | 感度分析：1 つのハイパーパラメータだけを動かす（7 パラメータ × 5 値 × 5 シード × 3 タスク）、最終エポックの EMA 教師 acc を報告。使用法：`python run_sensitivity.py --tasks AB,AC,BC`。 |
| `run_significance_test.py` | 有意性検定：FCCAN 対 各比較手法、3 タスク × 5 シードのシード別ペア検定、p 値と有意マークを出力（査読 M4）。 |
| `run_physics_15.py` | 周波数帯モジュールの物理的検証：15 モデル（3 タスク × 5 シード）、Phase 1 学習 + Phase 2 周波数帯摂動/エネルギー解析、`physics15/` に出力。使用法：`python run_physics_15.py --phase both`。 |
| `run_batch_template.py` | 一括実験テンプレート。コピーして `BASE` / `CONFIGS` を編集すれば、複数設定 × シードを一括実行できます（student/EMA acc を記録）。 |
| `experiment_ll_shortcut.py` | 低周波ショートカット仮説の検証：ソースでベースラインとオーグメントの 2 モデルを学習し、ターゲットテストセットに低/高周波の摂動を加えて、「モデルは低周波で分類し、オーグメントがショートカットを壊す」ことを確かめます。`results_ll_shortcut.txt` を出力。 |
| `measure_latency.py` | 推論時間の測定（論文 §4.10 効率分析）：全比較手法 + FCCAN のターゲットテストセットでの推論時間、パラメータ数、FLOPs。`measure_latency_results.csv` を出力。 |
| `measure_latency_trained.py` | 実学習済みの重みで FCCAN の推論レイテンシを測定（論文 §4.10 最終版プロトコル）。 |
| `parse_ablation_student.py` | アブレーションログから no_ema の Student 最終エポック指標を解析（査読 M2 データ）、`results_ablation_3tasks.txt` に追記。 |
| `_stats_adda_emdda.py` | ADDA / EM-DDA 比較実験の統計（第 10 エポック final-epoch で統一、論文プロトコルに一致）。 |
| `requirements.txt` | 依存リスト（torch、torchvision、numpy、scipy、matplotlib、scikit-learn 等）。 |

### 1.2 ルート：ドキュメントと結果ファイル

| ファイル | 用途 |
|---|---|
| `best.md` | 各タスクの最良設定、シード再測定、過去の最良記録（§9 再現設定の根拠）。 |
| `FCCAN_pipeline.md` | 手法全体のパイプライン説明（FEA 周波数帯強調 + CaCo + 低周波抑制）。 |
| `reference_notes.md` | 参考文献記録（番号付き引用マスターリスト）。 |
| `comparison_results_summary.md` | 比較実験結果のまとめ（論文表 1 のデータ）。 |
| `sensitivity_analysis_summary.md` | 感度分析結果のまとめ。 |
| `results_ablation_3tasks.txt` | メインアブレーション（8 モジュール）の結果。 |
| `results_ablation_basic.txt` | 初期の基礎アブレーション結果（履歴）。 |
| `results_comparison_all.csv` | 比較手法のシード別結果（有意性検定の入力データ）。 |
| `measure_latency_results.csv` / `measure_latency_results_server.csv` | ローカル / サーバーでの推論レイテンシ測定結果。 |
| `best.zip` / `comparison_experiments.rar` | 履歴バックアップアーカイブ（`best/`、`comparison_experiments/` の圧縮）。**任意、削除可**。 |
| `pytorch_model.bin` | 事前学習済み重みキャッシュ。**任意、削除可**（再現には不要）。 |
| `.gitignore` | Git 無視ルール（ログ、学習成果物、論文 PDF、IDE 等）。 |
| `.instructions.md` | Copilot プロジェクト指示（マルチエージェント研究支援設定）、コード実行とは無関係。 |

### 1.3 コアコードディレクトリ

| ディレクトリ / ファイル | 用途 |
|---|---|
| `models/fea_net.py` | モデル定義：`FEANet`（周波数帯強調メインモデル）、`FEANetBase`（基本版）、`Classifier`（分類ヘッド）。 |
| `trainers/source_trainer.py` | ソース学習（改良版 + 厳密ベースライン `--use_baseline 1`）。 |
| `trainers/target_trainer.py` | ターゲット学習：CaCo メインパス + ベースライン、EMA 教師ガイダンスのロジックを含む。 |
| `util/ang.py` | Batch ETF 角度均衡（ANG、クラス間角度の最大化）。 |
| `util/caco_loss.py` | CaCo カテゴリ対比損失。 |
| `util/data_utils.py` | データパスと共通ユーティリティ：タスクリスト、データセットディレクトリ対応、保存ディレクトリ。 |
| `util/diag_runtime.py` | DiagV2 ランタイム：グローバルコレクタ、各エポック診断フック、履歴保存。 |
| `util/diag_v2.py` | 学習の健康診断指標：混同行列、クラス別 recall、プロトタイプ余弦行列、特徴ノルム等。 |
| `util/em_loss.py` | エントロピー最小化（EM）損失ファミリー（SCW-LL ショートカット認識エントロピー重み付けを含む）。 |
| `util/energy_uda.py` | エネルギー整列：SCAL（自由エネルギー整列）+ SCON（スコア正規化）、Herath et al. ICCV 2023。 |
| `util/eval_utils.py` | 評価と診断レポート：`test()`、特徴診断、JSON/TXT 保存。 |
| `util/ll_strength_aug.py` | 低周波（LL）摂動オーグメンテーション、「明るさ=クラス」ショートカットを破壊。 |
| `util/lr_schedules.py` | 学習率スケジュール。 |
| `util/result_logger.py` | 統合結果エクスポート（`--save_result_txt`）。 |
| `util/tsne_utils.py` | t-SNE 可視化と学習ログ保存。 |
| `util/wavelet_recal.py` | ウェーブレット再較正（Haar DWT 重み構築、FEA-Net 周波数帯モジュールの中核）。 |

### 1.4 データと成果物

| ディレクトリ | 用途 |
|---|---|
| `datasets/` | データ（BOE / TMI / CELL の 3 ドメイン、人物単位分割）、構造は §4。 |
| `saves/` | 学習成果物：モデル重み（`source_encoder.pt`、`target_encoder.pt`、`classifier.pt`）、指標、診断履歴。自動生成、削除可。 |
| `results/` | t-SNE 図、学習ログ。自動生成、削除可。 |
| `best/` | 論文メインテーブルデータ：3 タスク × 5 シードの完全指標、t-SNE、`SUMMARY.txt`（`run_best_metrics.py` が生成）。 |
| `physics15/` | 周波数帯の物理的検証の成果物（`run_physics_15.py` が生成、gitignore 済み）。 |
| `review_results/` | 査読関連の成果物（gitignore 済み）。 |
| `docs/` | 研究ノート、査読記録、実験分析（履歴参照用；論文の重要結論はルートの `*_summary.md` に集約）。 |
| `figures/` | アーキテクチャ図のソース（`fccan_test.drawio`、draw.io で開く）。 |

### 1.5 比較実験（使い方）

`comparison_experiments/` は論文表 1 の比較手法の実装で、手法ごとに 1 ディレクトリ、共通のデータ/評価プロトコル（`common/`）を共有します。**実装を個別に読む必要はありません**。再現は次の 2 コマンドだけです。

```bash
python -m comparison_experiments.run_all --methods dann,mcc,emdda --tasks A-B,A-C --seeds 42,777
python run_comparison.py          # または全手法を一括実行
```

手法一覧（13 の DA 手法 + ベースライン）：`DANN`、`ADDA`、`CDAN`、`EM-DDA`、`MCC`、`FDA`、`CAN`、`SHOT`、`CAT`、`SVDNA`、`DAGCN`、`TVT`、`DaC`；加えて 3 つの source-only 下限（ResNet-18/50、VGG-16）と 1 つの oracle 参照行。詳細は `comparison_experiments/README.md`。

> 注：`FDA` は実装済みですが、クロスデバイス OCT では負の転移を示すため、論文の比較表には含めていません（コードは再現用に残置）。

> `BNM` / `TENT` / `DAN` / `DDC` は OCT クロスデバイスタスクで完全な負の転移（source-only 下限未満）を示したため、実験と論文から削除済みです。

### 1.6 説明対象外のディレクトリ

| ディレクトリ | 備考 |
|---|---|
| `ai_skills/` | 外部スキルパッケージ（指導教員の研究支援スキル）、コードと無関係、gitignore 済み。 |
| `paper/`（中国語論文）、`paper_ieee/`（英語論文） | 論文 LaTeX ソース、再現コードではない。 |
| `ppt/` | 発表 PPT の作業ディレクトリ、再現コードではない。 |
| `_ref_oct_dda/` | 公式リファレンス実装（OCT-DDA ベースライン、比較手法の位置合わせ用）、メインパイプライン外。 |

---

## 2. 環境要件

| 項目 | 要件 |
|---|---|
| Python | 3.9 ~ 3.11（3.10 推奨） |
| DL フレームワーク | PyTorch 2.0+（torchvision 含む） |
| GPU | NVIDIA GPU 推奨（CUDA）；CPU のみでも動くが遅い |
| OS | Windows / Linux / macOS |

---

## 3. インストール（約 10 分）

**3.1 仮想環境の作成**

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

**3.2 PyTorch のインストール**（[pytorch.org](https://pytorch.org/get-started/locally/) で GPU に合うコマンドを生成）

```bash
# NVIDIA GPU あり（CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# NVIDIA GPU なし（CPU のみ）
pip install torch torchvision
```

> torch を先にインストールし、その後に残りの依存を入れてください（バージョン競合の防止）。

**3.3 残りの依存をインストール**

```bash
pip install -r requirements.txt
```

**3.4 インストールの確認**

```bash
python -c "import torch, torchvision; print(torch.__version__)"
python tests/run_all.py        # すべて通れば環境は準備完了
```

---

## 4. データ準備

データは `datasets/` 以下に配置し、**train / val / test の各クラスを 1 フォルダ**にします。フォルダ名はクラス名（`AMD`、`DME`、`NORMAL`）である必要があります。

```
datasets/
├── BOE_split_by_person/          # ソース：BOE
│   ├── train/AMD/  train/DME/  train/NORMAL/
│   ├── val/AMD/     val/DME/     val/NORMAL/
│   └── test/AMD/    test/DME/    test/NORMAL/
├── TMIdata_split_by_person/      # ターゲット：TMI（train はラベルなし、フォルダのみ）
│   └── test/AMD/   test/DME/   test/NORMAL/
└── CELL_split_2025/              # 3 番目のドメイン：CELL（75% ラベルなしプロトコル）
    ├── train/...  test/...
```

> プログラムは `*_bg_removed`（背景除去）版ディレクトリがあれば優先し、なければ原画像を使用します。
> 論文の 3 タスクは `--cell_split CELL_split_2025` で統一しています。

### 4.1 元データの入手先

データセットの著作権は元論文の著者に帰属します。本リポジトリには生画像は含まれていません。下記の原本から入手し、上記のディレクトリ構成に整理してください。

| データセット | 元データの出典 | 入手方法 |
|---|---|---|
| BOE（Dataset A） | Srinivasan et al., *Biomedical Optics Express* 5(10), 2014（Duke SD-OCT、45 名の患者） | 公式ページ <https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm>；直接ダウンロード <http://www.duke.edu/~sf59/Datasets/2014_BOE_Srinivasan.zip>（研究・教育目的のみ、商用再配布禁止、論文の引用必須） |
| TMI（Dataset B） | 同論文の Macular Dataset-Heidelberg（Noor Eye Hospital, Tehran、148 名の患者） | 公開ホストなし。著者へ連絡して入手：pratul.srinivasan@gmail.com または sina.farsiu@duke.edu |
| CELL（Dataset C） | Kermany et al., *Cell* 172(5), 2018 の OCT2017（本リポジトリは DRUSEN→AMD、DME、NORMAL の 3 クラスを使用し、CNV は破棄） | Mendeley Data <https://data.mendeley.com/datasets/rscbjbr9sj/2>（CC BY 4.0）；OCT のみは `OCT2017.tar.gz`（5.4 GB）をダウンロード：<https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded> |

### 4.2 データの処理と配置場所

元データ入手後、以下の手順で整理し、最終的にプロジェクトルートの `datasets/` 配下に配置してください。

**BOE**
1. `2014_BOE_Srinivasan.zip` を解凍し、ボリュームデータを患者ごとのフォルダ（クラス：`AMD` / `DME` / `NORMAL`）に JPG として整理し、`BOE_dataset_subject_JPG/` とします。
2. 患者単位で 50% 訓練 / 25% 検証 / 25% テストに分割します（参考実装：`_ref_oct_dda/BOE_data_split_by_person.py`）。
3. 分割結果を `datasets/BOE_split_by_person/` に配置します（構造は §4）。

**TMI**
1. Srinivasan et al. 2014 の著者へ Macular Dataset-Heidelberg の提供を依頼してください。**本リポジトリはこのデータを提供・再配布しません。リポジトリの著者へ要求しないでください。**
2. 患者ごとに `TMIdata_subject_JPG/` として整理し、患者単位で分割します（ターゲット時：約 75% 訓練 / 25% テスト。参考実装：`_ref_oct_dda/TMI_data_split_by_person.py`）。
3. 分割結果を `datasets/TMIdata_split_by_person/` に配置します。

**CELL**
1. Mendeley から `OCT2017.tar.gz` をダウンロードして解凍し、`CNV / DME / DRUSEN / NORMAL` の 4 クラスフォルダを得ます。
2. `CNV` を削除し、`DRUSEN` を `AMD` にリネームして、3 クラスの `OCT2017/` とします。
3. ファイル名の患者番号（`(クラス)-(患者ID)-(画像番号)` 形式）に基づいて患者単位で分割します（クラスごと 356 訓練 / 200 検証 / 186 テスト、計 742 枚/クラス）。
4. 分割結果を `datasets/CELL_split_2025/` に配置します（論文の全タスクは `--cell_split CELL_split_2025` を使用）。

---

## 5. クイックスタート

```bash
# 最小コマンド：デフォルト設定で BOE -> TMI を実行
python main.py --only BOE->TMI
```

実行後、`saves/` にモデル、`results/` に t-SNE 図と指標が生成され、ターミナルにクラス別の正解率が表示されます。

---

## 6. よく使うコマンド

```bash
# ソース/ターゲット指定（A=BOE, B=TMI, C=CELL の省略可）
python main.py --source BOE --target TMI
python main.py --only A->C

# 複数回実行して平均（論文は 5 シード、各タスクの best 設定は §9）
python main.py --only BOE->TMI -i 5

# 他の転移タスク
python main.py --only BOE->CELL
python main.py --only TMI->BOE

# ソースのみ、転移なし（ソース自体の性能評価）
python main.py --only BOE->TMI -l 0

# 厳密な元のベースライン（論文の reference ベースライン）
python main.py --only BOE->TMI --use_baseline 1

# 乱数シード固定（再現用）
python main.py --only BOE->TMI --seed 777
```

---

## 7. パラメータ解説

> 完全なリストは `python main.py --help` で確認できます。以下はよく使うスイッチを機能別にまとめたものです。

### 7.1 タスクとデータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `-s / --source` | `BOE` | ソースデータセット：`BOE` / `CELL` / `TMI` |
| `-t / --target` | `TMI` | ターゲットデータセット |
| `--only` | `BOE->TMI` | 指定タスクのみ実行、例：`BOE->TMI` または `A->B` |
| `--batch_src` / `--batch_tgt` | `16` | ソース/ターゲットのバッチサイズ、OOM なら小さく |
| `--input_size` | `256` | 入力解像度、メモリが許せば 448/512 に上げて詳細を保持 |
| `-i / --iterations` | `1` | 繰り返し回数、論文は 5 回平均 |

### 7.2 学習フロー

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `-l / --transferlearning` | `1` | `1`=ドメイン適応転移、`0`=ソースのみ |
| `-es / --epochs_src` | `5` | ソース学習エポック（best：A-B=5 / A-C=4 / B-C=8） |
| `-et / --epochs_tgt` | `30` | ターゲット学習エポック（best：A-B=8 / A-C=15 / B-C=15） |
| `--use_baseline` | `0` | `1`=厳密な元のベースライン |
| `--cell_split` | `CELL_split` | データ分割プロトコル（論文は `CELL_split_2025`） |
| `--save_which` | `ema` | EMA 教師（`ema`）または Student（`student`）の重みを保存/報告 |

### 7.3 モデル構造（FEA-Net）

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--use_fea_net` | `1` | `1`=FEA-Net メインモデル（ウェーブレット周波数強調） |
| `--use_hf_comp` | `1` | 高周波補償ブランチ |
| `--use_msw_sa` | `1` | ウェーブレット空間アテンション |
| `--wrb_alpha` / `--wrb_lambda` | `0.4` / `0.3` | ウェーブレット残差ブロックのパラメータ |

### 7.4 損失モジュール

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--lambda_caco` | `0.1` | CaCo 対比損失の重み |
| `--caco_key_conf` | `0.95` | CaCo は高信頼サンプルのみ key に使用、0=フィルタなし |
| `--lambda_em` | `1.0` | エントロピー最小化の重み、0=オフ |
| `--scw_ll` | `1.0` | 低周波ショートカット認識エントロピー重み付け |
| `--lambda_llinv` | `1.0` | 低周波不変性の一貫性（自己教師あり） |
| `--ema_guide_caco` | `1.0` | EMA 教師が CaCo をガイド、疑似ラベルを平滑化 |
| `--ema_guide_warmup` | `10` | EMA ガイダンスのウォームアップエポック |
| `--lambda_batch_ang` | `0.5` | Batch ETF 角度均衡 |
| `--lambda_src` | `0.1` | ターゲット段階でのソース忘却防止の重み |
| `--use_energy_uda` | `1` | エネルギー整列 SCAL+SCON |

### 7.5 データオーグメンテーション（低周波ショートカット抑制）

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--src_ll_aug` | `1.0` | ソース低周波（LL）エネルギー摂動、「明るさ=クラス」ショートカットを破壊 |
| `--src_ll_alpha` | `1.0` | 摂動の大きさ |
| `--src_ll_prob` | `0.5` | 摂動を適用する確率 |

### 7.6 診断と再現

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--use_diag_v2` | `1` | 各エポックで健康診断指標を出力 |
| `--seed` | `3407` | 乱数シード、`-1`=未設定 |

---

## 8. セルフチェック

```bash
python tests/run_all.py
```

対象：全ユーティリティモジュールのコア関数、モデルの forward 形状、学習ループがダミーデータで 1 エポック回るか。**すべて通れば = コードが壊れていない・環境が準備完了**。事前学習重みのダウンロードに失敗した場合は該当テストを自動スキップします。

---

## 9. 論文再現設定

> 以下は論文（`best/SUMMARY.txt`）の各タスク最良設定です。この通りに実行すれば論文の結果を再現できます（5 シード mean ± std、EMA 教師固定エポック精度、`--save_which ema`）。

### 9.1 BOE→TMI（5 シード EMA = 0.9558 ± 0.0153）

```bash
python main.py --only BOE->TMI -es 5 -et 8 --seed 42  -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 123 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 777 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 2024 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 3407 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
```

### 9.2 BOE→CELL（5 シード EMA = 0.8774 ± 0.0245）

```bash
python main.py --only BOE->CELL -es 4 -et 15 --seed 42  -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 123 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 777 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 2024 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 3407 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
```

### 9.3 TMI→CELL（5 シード EMA = 0.9212 ± 0.0101）

```bash
python main.py --only TMI->CELL -es 8 -et 15 --seed 42  -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 123 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 777 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 2024 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 3407 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
```

### 9.4 設定まとめ表

| タスク | es | et | 主要パラメータ | 5 シード EMA mean ± std |
|---|---|---|---|---|
| BOE→TMI | 5 | 8 | `--src_ll_prob 0.7 --src_ll_alpha 2.0` | **0.9558 ± 0.0153** |
| BOE→CELL | 4 | 15 | `--lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01` | **0.8774 ± 0.0245** |
| TMI→CELL | 8 | 15 | `--lambda_caco 0.01 --lambda_batch_ang 0.5` | **0.9212 ± 0.0101** |

---

## 10. 実験スクリプト解説

論文の各解析実験の再現・実行方法（詳細は各スクリプトのヘッダコメント参照）：

| 実験 | コマンド |
|---|---|
| メインアブレーション（8 モジュール） | `python run_ablation.py --phase main` |
| 追加アブレーション（no_ema / no_fea_ll / oracle / src_only_fea） | `python run_ablation.py --phase extra` |
| 感度分析（全タスク） | `python run_sensitivity.py` |
| 感度分析（A-B のみ） | `python run_sensitivity.py --tasks AB` |
| 比較手法の一括実行 | `python run_comparison.py` |
| 有意性検定 | `python run_significance_test.py` |
| メインテーブルデータ（15 runs + t-SNE） | `python run_best_metrics.py` |
| 周波数帯の物理的検証 | `python run_physics_15.py --phase both` |
| 低周波ショートカット検証 | `python experiment_ll_shortcut.py` |
| 推論時間測定（全手法） | `python measure_latency.py` |
| 推論時間測定（学習済み重み） | `python measure_latency_trained.py` |

> これらの一括スクリプトは `--seeds` / `--tasks` / `--skip-existing` で作業量を調整できます。`--skip-existing` はログが既にある run をスキップします（再開対応）。

---

## 11. FAQ

**Q1: `No module named 'torch'` エラー？**
仮想環境が有効でないか、torch が未インストール。§3 に戻り、`pip list` に torch があるか確認してください。

**Q2: CUDA / GPU 関連エラー？**
- GPU なし：CPU 版 torch を入れ直し（§3.2 参照）。
- GPU エラー：NVIDIA ドライバと CUDA バージョンの一致を確認。

**Q3: メモリ不足（OOM）？**
`--batch_src` / `--batch_tgt`（例：`16 → 4`）と `--input_size` を小さくしてください。

**Q4: `results_*.txt` / `diagnosis_results.txt` とは？**
自動生成される学習診断ログです。削除しても動作に影響しません。

**Q5: `saves/` の `.pt` ファイルとは？**
学習済みモデル重みです：`source_encoder.pt`（ソース）、`target_encoder.pt`（ターゲット）、`classifier.pt`（分類ヘッド）。チェックポイント/論文解析に使用します。

**Q6: 論文の結果を再現するには？**
§9 の各タスク best 設定をシード別に実行（5 シードで mean ± std）、または `python run_best_metrics.py` を直接実行してください。

---

## 12. ライセンスと引用

本プロジェクトは MIT ライセンスで公開しています（[LICENSE](LICENSE) 参照）。

本プロジェクトは研究用コードで、DAGCN reference ベースラインを改良したものです。コア手法：ウェーブレット周波数強調（FEA）+ クラス対比（CaCo）+ 低周波ショートカット抑制。本コードを使用する場合は論文を引用してください（採録後に文献番号を追記予定）。

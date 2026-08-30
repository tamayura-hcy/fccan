# FCCAN

**Frequency-aware Contrastive Cross-domain Adaptation Network for Unsupervised Domain Adaptation of OCT Images**

**Keywords:** Unsupervised Domain Adaptation · OCT · Optical Coherence Tomography · Medical Imaging · Cross-Device · Frequency-Domain · Contrastive Learning · PyTorch

> 🌐 Switch language: **[English](README.en.md)** · **[日本語](README.ja.md)** · **[中文](README.md)**

FCCAN does unsupervised domain adaptation for OCT images. A model trained on a labeled source domain (e.g., BOE) is transferred to unlabeled target domains (e.g., TMI, CELL) for stable **AMD / DME / NORMAL** classification.

The framework has no adversarial training (no discriminator, no gradient reversal). It rests on two lines:

- **Frequency-band awareness**: FEA-Net band-enhanced backbone + low-frequency shortcut suppression;
- **Stable class-level alignment**: EMA-teacher-guided category contrast + energy normalization + angular balance.

5-seed mean accuracy (EMA teacher, fixed epoch) on three cross-device scenarios: **BOE→TMI 95.6% · BOE→CELL 87.7% · TMI→CELL 92.1%**.

> Naming: **FCCAN** is the full method (the overall framework); **FEA-Net** (Frequency-Enhanced Attention Network) is its band-enhanced backbone.

---

## Table of Contents

1. [Directory Structure & File Reference](#1-directory-structure--file-reference)
2. [Environment Requirements](#2-environment-requirements)
3. [Installation](#3-installation)
4. [Prepare Data](#4-prepare-data)
5. [Quick Start](#5-quick-start)
6. [Common Commands](#6-common-commands)
7. [Parameter Reference](#7-parameter-reference)
8. [Run Self-Checks](#8-run-self-checks)
9. [Reproduction Config](#9-reproduction-config)
10. [Experiment Scripts](#10-experiment-scripts)
11. [FAQ](#11-faq)
12. [License & Citation](#12-license--citation)

---

## 1. Directory Structure & File Reference

### 1.1 Root: Core Scripts

| File | Purpose |
|---|---|
| `main.py` | The single training entry point. Source training, target transfer, and evaluation all go through it, with every module (FEA-Net, CaCo, EM, ANG, EMA teacher, low-frequency augmentation) and every hyperparameter (§7). |
| `repro_seeds.py` | Fixed seed set (middle 10 stable seeds). The paper, ablations, and sensitivity analyses share the same seeds so results are reproducible. |
| `run_best_metrics.py` | Runs best config × 5 seeds on three tasks (15 runs), writes full 10 metrics, t-SNE, and per-epoch diagnostics to `best/` (source of the paper's main table). Usage: `python run_best_metrics.py`. |
| `run_comparison.py` | Batch runner for comparison methods: runs each method in `comparison_experiments/` × tasks × seeds, results to `comparison_experiments/results/`. Usage: `python run_comparison.py`. |
| `run_ablation.py` | Unified ablation entry. `--phase main` runs the 8 w/o-module leave-one-out ablations (paper ablation table); `--phase extra` runs no_ema, no_fea_ll, oracle, src_only_fea. Usage: `python run_ablation.py --phase main`. |
| `run_sensitivity.py` | Sensitivity analysis: one hyperparameter at a time (7 params × 5 values × 5 seeds × 3 tasks), reports final-epoch EMA-teacher acc. Usage: `python run_sensitivity.py --tasks AB,AC,BC`. |
| `run_significance_test.py` | Significance test: FCCAN vs each comparison method, paired per-seed acc over 3 tasks × 5 seeds, outputs p-values and significance marks (reviewer M4). |
| `run_physics_15.py` | Physical proof of the frequency-band modules: 15 models (3 tasks × 5 seeds), Phase 1 train + Phase 2 band perturbation/energy analysis, outputs to `physics15/`. Usage: `python run_physics_15.py --phase both`. |
| `run_batch_template.py` | Batch experiment template. Copy it, edit `BASE` / `CONFIGS`, and it runs many configs × seeds (logs student/EMA acc). |
| `experiment_ll_shortcut.py` | Low-frequency shortcut verification: trains baseline and augmented source models, perturbs low/high frequencies on the target test set, and checks whether models classify via low frequency and whether augmentation breaks that shortcut. Outputs `results_ll_shortcut.txt`. |
| `measure_latency.py` | Inference-time measurement (paper §4.10 efficiency analysis): inference time, parameters, and FLOPs of all comparison methods + FCCAN on the target test set. Outputs `measure_latency_results.csv`. |
| `measure_latency_trained.py` | Measures FCCAN inference latency with real trained weights (final paper §4.10 protocol). |
| `parse_ablation_student.py` | Parses Student final-epoch metrics for the no_ema plan from ablation logs (reviewer M2 data), appends to `results_ablation_3tasks.txt`. |
| `_stats_adda_emdda.py` | ADDA / EM-DDA comparison statistics (unified epoch-10 final-epoch protocol, consistent with the paper). |
| `requirements.txt` | Dependency list (torch, torchvision, numpy, scipy, matplotlib, scikit-learn, etc.). |

### 1.2 Root: Docs & Result Files

| File | Purpose |
|---|---|
| `best.md` | Best configs per task, seed re-runs, historical bests (basis of the §9 reproduction config). |
| `FCCAN_pipeline.md` | Overall method pipeline (FEA band enhancement + CaCo + low-frequency suppression). |
| `reference_notes.md` | Literature references (numbered citation master list). |
| `comparison_results_summary.md` | Comparison experiment results (paper Table 1 data). |
| `sensitivity_analysis_summary.md` | Sensitivity analysis results. |
| `results_ablation_3tasks.txt` | Main ablation (8 modules) results. |
| `results_ablation_basic.txt` | Early basic ablation results (historical). |
| `results_comparison_all.csv` | Per-seed comparison results (input of the significance test). |
| `measure_latency_results.csv` / `measure_latency_results_server.csv` | Local / server inference latency results. |
| `best.zip` / `comparison_experiments.rar` | Historical backup archives of `best/` and `comparison_experiments/`. **Optional, safe to delete.** |
| `pytorch_model.bin` | Pretrained weight cache. **Optional, safe to delete** (not needed for reproduction). |
| `.gitignore` | Git ignore rules (logs, training artifacts, paper PDFs, IDE files, etc.). |
| `.instructions.md` | Copilot project instructions (multi-agent research assistant config), unrelated to running the code. |

### 1.3 Core Code Directories

| Directory / File | Purpose |
|---|---|
| `models/fea_net.py` | Model definitions: `FEANet` (band-enhanced main model), `FEANetBase` (base variant), `Classifier` (classification head). |
| `trainers/source_trainer.py` | Source training (improved version + strict baseline `--use_baseline 1`). |
| `trainers/target_trainer.py` | Target training: CaCo main path + baseline, including EMA-teacher guidance logic. |
| `util/ang.py` | Batch ETF angular balance (ANG, inter-class angle maximization). |
| `util/caco_loss.py` | CaCo category contrastive loss. |
| `util/data_utils.py` | Data paths and common utils: task list, dataset directory mapping, save directories. |
| `util/diag_runtime.py` | DiagV2 runtime: global collector, per-epoch diagnostic hooks, history persistence. |
| `util/diag_v2.py` | Training health metrics: confusion matrix, per-class recall, prototype cosine matrix, feature norms, etc. |
| `util/em_loss.py` | Entropy minimization (EM) loss family (incl. SCW-LL shortcut-aware entropy weighting). |
| `util/energy_uda.py` | Energy alignment: SCAL (free-energy alignment) + SCON (score normalization), Herath et al. ICCV 2023. |
| `util/eval_utils.py` | Evaluation and diagnostic reports: `test()`, feature diagnostics, JSON/TXT persistence. |
| `util/ll_strength_aug.py` | Low-frequency (LL) perturbation augmentation, breaks the "brightness = class" shortcut. |
| `util/lr_schedules.py` | Learning-rate schedules. |
| `util/result_logger.py` | Unified result export (`--save_result_txt`). |
| `util/tsne_utils.py` | t-SNE visualization and training-log persistence. |
| `util/wavelet_recal.py` | Wavelet recalibration (Haar DWT weight construction; core of the FEA-Net band module). |

### 1.4 Data & Artifacts

| Directory | Purpose |
|---|---|
| `datasets/` | Data (BOE / TMI / CELL domains, split by person), structure in §4. |
| `saves/` | Training artifacts: model weights (`source_encoder.pt`, `target_encoder.pt`, `classifier.pt`), metrics, diagnostics. Auto-generated, deletable. |
| `results/` | t-SNE plots, training logs. Auto-generated, deletable. |
| `best/` | Paper main-table data: 3 tasks × 5 seeds full metrics, t-SNE, `SUMMARY.txt` (generated by `run_best_metrics.py`). |
| `physics15/` | Frequency-band physical proof artifacts (generated by `run_physics_15.py`, gitignored). |
| `review_results/` | Reviewer-related artifacts (gitignored). |
| `docs/` | Research notes, review records, experiment analyses (historical reference; key paper conclusions are summarized in the root `*_summary.md` files). |
| `figures/` | Architecture diagram sources (`fccan_test.drawio`, open with draw.io). |

### 1.5 Comparison Experiments (How to Use)

`comparison_experiments/` holds the paper Table 1 comparison methods, one directory per method, sharing the same data/evaluation protocol (`common/`). **You do not need to read each implementation.** Two commands reproduce them:

```bash
python -m comparison_experiments.run_all --methods dann,mcc,emdda --tasks A-B,A-C --seeds 42,777
python run_comparison.py          # or run everything in one batch
```

Methods (13 DA methods + baselines): `DANN`, `ADDA`, `CDAN`, `EM-DDA`, `MCC`, `FDA`, `CAN`, `SHOT`, `CAT`, `SVDNA`, `DAGCN`, `TVT`, `DaC`; plus 3 source-only lower bounds (ResNet-18/50, VGG-16) and 1 oracle reference row. See `comparison_experiments/README.md`.

> `BNM` / `TENT` / `DAN` / `DDC` show fully negative transfer on OCT cross-device tasks (below the source-only lower bound) and have been removed from the experiments and the paper.

### 1.6 Directories Not Documented

| Directory | Note |
|---|---|
| `ai_skills/` | External skill package (supervisor research-assistant skills), unrelated to the code, gitignored. |
| `paper/` (Chinese paper), `paper_ieee/` (English paper) | Paper LaTeX sources, not reproduction code. |
| `ppt/` | Presentation PPT working directory, not reproduction code. |
| `_ref_oct_dda/` | Official reference implementation (OCT-DDA baseline, used to align comparison methods), not part of the main pipeline. |

---

## 2. Environment Requirements

| Item | Requirement |
|---|---|
| Python | 3.9 ~ 3.11 (3.10 recommended) |
| DL Framework | PyTorch 2.0+ (with torchvision) |
| GPU | NVIDIA GPU recommended (CUDA); CPU-only works but is slow |
| OS | Windows / Linux / macOS |

---

## 3. Installation (~10 min)

**3.1 Create a virtual environment**

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

**3.2 Install PyTorch** (pick your build at [pytorch.org](https://pytorch.org/get-started/locally/))

```bash
# With an NVIDIA GPU (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Without an NVIDIA GPU (CPU only)
pip install torch torchvision
```

> Install torch first, then the remaining dependencies, to avoid version conflicts.

**3.3 Install the remaining dependencies**

```bash
pip install -r requirements.txt
```

**3.4 Verify the installation**

```bash
python -c "import torch, torchvision; print(torch.__version__)"
python tests/run_all.py        # all OK means the environment is ready
```

---

## 4. Prepare Data

Place data under `datasets/` with **one folder per class in train / val / test**; folder names must be the class names (`AMD`, `DME`, `NORMAL`).

```
datasets/
├── BOE_split_by_person/          # Source: BOE
│   ├── train/AMD/  train/DME/  train/NORMAL/
│   ├── val/AMD/     val/DME/     val/NORMAL/
│   └── test/AMD/    test/DME/    test/NORMAL/
├── TMIdata_split_by_person/      # Target: TMI (train unlabeled, folders only)
│   └── test/AMD/   test/DME/   test/NORMAL/
└── CELL_split_2025/              # Third domain: CELL (75% unlabeled protocol)
    ├── train/...  test/...
```

> The program prefers `*_bg_removed` (background-removed) directories when they exist, otherwise the originals.
> All three paper tasks use `--cell_split CELL_split_2025`.

### 4.1 Original Dataset Sources

The datasets belong to their original authors; this repository contains no raw images. Download from the original sources below and reorganize them into the structure above.

| Dataset | Original source | How to obtain |
|---|---|---|
| BOE (Dataset A) | Srinivasan et al., *Biomedical Optics Express* 5(10), 2014 (Duke SD-OCT, 45 patients) | Official page <https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm>; direct download <http://www.duke.edu/~sf59/Datasets/2014_BOE_Srinivasan.zip> (research/education use only, commercial redistribution prohibited, cite the paper) |
| TMI (Dataset B) | Macular Dataset-Heidelberg of the same paper (Noor Eye Hospital, Tehran, 148 patients) | Not publicly hosted; contact the authors: pratul.srinivasan@gmail.com or sina.farsiu@duke.edu |
| CELL (Dataset C) | OCT2017 of Kermany et al., *Cell* 172(5), 2018 (this repo uses DRUSEN→AMD, DME, NORMAL; CNV discarded) | Mendeley Data <https://data.mendeley.com/datasets/rscbjbr9sj/2> (CC BY 4.0); OCT-only download `OCT2017.tar.gz` (5.4 GB): <https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded> |

### 4.2 Processing and Placement

After downloading the originals, follow the steps below; the final data go under the project root `datasets/`.

**BOE**
1. Unzip `2014_BOE_Srinivasan.zip` and organize the volumetric scans into per-patient JPG folders with classes `AMD` / `DME` / `NORMAL` (call it `BOE_dataset_subject_JPG/`).
2. Split by patient (50% train / 25% val / 25% test); see the reference implementation `_ref_oct_dda/BOE_data_split_by_person.py`.
3. Place the result at `datasets/BOE_split_by_person/` (structure in §4).

**TMI**
1. Request the Macular Dataset-Heidelberg from the authors of Srinivasan et al. 2014. **This repository does not provide or redistribute it; do not ask the repository authors for it.**
2. Organize it by patient as `TMIdata_subject_JPG/` and split by patient (as target: ~75% train / 25% test); see `_ref_oct_dda/TMI_data_split_by_person.py`.
3. Place the result at `datasets/TMIdata_split_by_person/`.

**CELL**
1. Download `OCT2017.tar.gz` from Mendeley and unzip it into the four class folders `CNV / DME / DRUSEN / NORMAL`.
2. Delete `CNV` and rename `DRUSEN` to `AMD`, giving a three-class `OCT2017/`.
3. Split by the patient ID encoded in the file name (`(class)-(patient ID)-(image index)`): 356 train / 200 val / 186 test per class (742 per class in total).
4. Place the result at `datasets/CELL_split_2025/` (all paper tasks use `--cell_split CELL_split_2025`).

---

## 5. Quick Start

```bash
# Minimal command: run BOE -> TMI with defaults
python main.py --only BOE->TMI
```

When it finishes, models are under `saves/`, t-SNE plots and metrics under `results/`, and per-class accuracy is printed.

---

## 6. Common Commands

```bash
# Pick source/target (A=BOE, B=TMI, C=CELL shorthand)
python main.py --source BOE --target TMI
python main.py --only A->C

# Run several times and average (paper uses 5 seeds; see §9 for best configs)
python main.py --only BOE->TMI -i 5

# Other transfer tasks
python main.py --only BOE->CELL
python main.py --only TMI->BOE

# Source-only, no transfer (evaluate source ability itself)
python main.py --only BOE->TMI -l 0

# Strict original reference baseline
python main.py --only BOE->TMI --use_baseline 1

# Fix the random seed (reproducibility)
python main.py --only BOE->TMI --seed 777
```

---

## 7. Parameter Reference

> The full list is always available via `python main.py --help`. Below are the common switches grouped by function.

### 7.1 Task & Data

| Param | Default | Description |
|---|---|---|
| `-s / --source` | `BOE` | Source dataset: `BOE` / `CELL` / `TMI` |
| `-t / --target` | `TMI` | Target dataset |
| `--only` | `BOE->TMI` | Run one transfer task only, e.g. `BOE->TMI` or `A->B` |
| `--batch_src` / `--batch_tgt` | `16` | Source/target batch sizes; lower if OOM |
| `--input_size` | `256` | Input resolution; larger (448/512) keeps detail if memory allows |
| `-i / --iterations` | `1` | Repeat count; paper averages 5 |

### 7.2 Training Flow

| Param | Default | Description |
|---|---|---|
| `-l / --transferlearning` | `1` | `1`=transfer, `0`=source-only |
| `-es / --epochs_src` | `5` | Source epochs (best: A-B=5 / A-C=4 / B-C=8) |
| `-et / --epochs_tgt` | `30` | Target epochs (best: A-B=8 / A-C=15 / B-C=15) |
| `--use_baseline` | `0` | `1`=strict original reference baseline |
| `--cell_split` | `CELL_split` | Data split protocol (paper uses `CELL_split_2025`) |
| `--save_which` | `ema` | Save/report EMA teacher (`ema`) or Student (`student`) weights |

### 7.3 Model Architecture (FEA-Net)

| Param | Default | Description |
|---|---|---|
| `--use_fea_net` | `1` | `1`=main FEA-Net model (wavelet frequency enhancement) |
| `--use_hf_comp` | `1` | High-frequency compensation branch |
| `--use_msw_sa` | `1` | Wavelet spatial attention |
| `--wrb_alpha` / `--wrb_lambda` | `0.4` / `0.3` | Wavelet residual block params |

### 7.4 Loss Modules

| Param | Default | Description |
|---|---|---|
| `--lambda_caco` | `0.1` | CaCo contrastive weight |
| `--caco_key_conf` | `0.95` | CaCo uses only high-confidence samples as keys; 0=no filtering |
| `--lambda_em` | `1.0` | Entropy minimization weight, 0=off |
| `--scw_ll` | `1.0` | Low-frequency shortcut-aware entropy weighting |
| `--lambda_llinv` | `1.0` | Low-frequency invariance consistency (self-supervised) |
| `--ema_guide_caco` | `1.0` | EMA teacher guides CaCo, smooths pseudo-labels |
| `--ema_guide_warmup` | `10` | EMA guiding warmup epochs |
| `--lambda_batch_ang` | `0.5` | Batch ETF angular balance |
| `--lambda_src` | `0.1` | Source anti-forgetting weight in the target phase |
| `--use_energy_uda` | `1` | Energy alignment SCAL+SCON |

### 7.5 Data Augmentation (low-frequency shortcut suppression)

| Param | Default | Description |
|---|---|---|
| `--src_ll_aug` | `1.0` | Source low-frequency (LL) energy perturbation, breaks the "brightness=class" shortcut |
| `--src_ll_alpha` | `1.0` | Perturbation magnitude |
| `--src_ll_prob` | `0.5` | Perturbation probability |

### 7.6 Diagnostics & Reproducibility

| Param | Default | Description |
|---|---|---|
| `--use_diag_v2` | `1` | Print health metrics each epoch |
| `--seed` | `3407` | Random seed, `-1`=unset |

---

## 8. Run Self-Checks

```bash
python tests/run_all.py
```

Covers: core util functions, model forward shapes, and training loops on fake data for 1 epoch. **All green = code is intact and the environment is ready.** Tests auto-skip when pretrained weights cannot be downloaded.

---

## 9. Reproduction Config

> Below are the paper's best configs per task (`best/SUMMARY.txt`). Run them to reproduce the reported results (5-seed mean ± std, EMA-teacher fixed-epoch accuracy, `--save_which ema`).

### 9.1 BOE→TMI (5-seed EMA = 0.9558 ± 0.0153)

```bash
python main.py --only BOE->TMI -es 5 -et 8 --seed 42  -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 123 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 777 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 2024 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
python main.py --only BOE->TMI -es 5 -et 8 --seed 3407 -i 1 --cell_split CELL_split_2025 --src_ll_prob 0.7 --src_ll_alpha 2.0 --save_which ema
```

### 9.2 BOE→CELL (5-seed EMA = 0.8774 ± 0.0245)

```bash
python main.py --only BOE->CELL -es 4 -et 15 --seed 42  -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 123 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 777 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 2024 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
python main.py --only BOE->CELL -es 4 -et 15 --seed 3407 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01 --save_which ema
```

### 9.3 TMI→CELL (5-seed EMA = 0.9212 ± 0.0101)

```bash
python main.py --only TMI->CELL -es 8 -et 15 --seed 42  -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 123 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 777 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 2024 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
python main.py --only TMI->CELL -es 8 -et 15 --seed 3407 -i 1 --cell_split CELL_split_2025 --lambda_caco 0.01 --lambda_batch_ang 0.5 --save_which ema
```

### 9.4 Config Summary

| Task | es | et | Key params | 5-seed EMA mean ± std |
|---|---|---|---|---|
| BOE→TMI | 5 | 8 | `--src_ll_prob 0.7 --src_ll_alpha 2.0` | **0.9558 ± 0.0153** |
| BOE→CELL | 4 | 15 | `--lambda_caco 0.01 --lambda_batch_ang 0.001 --alpha_scon 0.01` | **0.8774 ± 0.0245** |
| TMI→CELL | 8 | 15 | `--lambda_caco 0.01 --lambda_batch_ang 0.5` | **0.9212 ± 0.0101** |

---

## 10. Experiment Scripts

How to run each analysis in the paper (see the header comments of each script for details):

| Experiment | Command |
|---|---|
| Main ablation (8 w/o modules) | `python run_ablation.py --phase main` |
| Extra ablation (no_ema / no_fea_ll / oracle / src_only_fea) | `python run_ablation.py --phase extra` |
| Sensitivity analysis (all tasks) | `python run_sensitivity.py` |
| Sensitivity analysis (A-B only) | `python run_sensitivity.py --tasks AB` |
| Comparison methods batch | `python run_comparison.py` |
| Significance test | `python run_significance_test.py` |
| Main-table data (15 runs + t-SNE) | `python run_best_metrics.py` |
| Frequency-band physical proof | `python run_physics_15.py --phase both` |
| Low-frequency shortcut verification | `python experiment_ll_shortcut.py` |
| Inference-time measurement (all methods) | `python measure_latency.py` |
| Inference-time measurement (trained weights) | `python measure_latency_trained.py` |

> These batch scripts accept `--seeds` / `--tasks` / `--skip-existing` to trim the workload; `--skip-existing` skips runs whose logs already exist (resume support).

---

## 11. FAQ

**Q1: `No module named 'torch'`?**
The virtual environment is not activated or torch is not installed. Go back to §3 and confirm torch is in `pip list`.

**Q2: CUDA / GPU-related errors?**
- No GPU: reinstall the CPU-only torch (see §3.2).
- GPU errors: check that the NVIDIA driver and CUDA version match.

**Q3: Out of memory (OOM)?**
Reduce `--batch_src` / `--batch_tgt` (e.g. `16 → 4`) and `--input_size`.

**Q4: What are `results_*.txt` / `diagnosis_results.txt`?**
Auto-generated training diagnostic logs; safe to delete, they do not affect training.

**Q5: What are the `.pt` files in `saves/`?**
Trained model weights: `source_encoder.pt` (source), `target_encoder.pt` (target), `classifier.pt` (head). Used for checkpointing and analysis.

**Q6: How to reproduce the paper results?**
Run the per-seed best configs in §9 (5 seeds, mean ± std), or simply `python run_best_metrics.py`.

---

## 12. License & Citation

This is research code, improved from the DAGCN reference baseline. Core method: wavelet frequency enhancement (FEA) + category contrast (CaCo) + low-frequency shortcut suppression. For citation, please contact the author.

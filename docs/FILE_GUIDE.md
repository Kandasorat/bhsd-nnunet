# File Guide

This document explains what each current repository file or directory is for, with a focus on the reproducible BHSD nnU-Net research pipeline.

## Top Level

### `.git/`

Local Git metadata directory. Not part of the research pipeline itself.

### `.github/`

GitHub-related helper files.

- `.github/copilot-instructions.md`
  Legacy repository guidance file. It currently describes an older BHSD sanity-check script workflow and is no longer aligned with the current nnU-Net research pipeline. Treat it as outdated unless it is rewritten later.

### `analysis/`

Scripts that summarize experiment outputs into comparison tables and report-ready artifacts.

- `analysis/__init__.py`
  Marks `analysis` as a Python package.

- `analysis/README.md`
  Short description of the purpose of the analysis module.

- `analysis/build_report.py`
  High-level report builder. Runs model comparison, table generation, figure generation, and optional statistical comparisons from a combined case-metrics CSV.

- `analysis/collect_results.py`
  Merges per-experiment case-level metric CSV files from `results/` into one combined CSV.

- `analysis/compare_models.py`
  Produces model-level summary statistics such as mean, standard deviation, median, min, and max for a chosen metric.

- `analysis/generate_figures.py`
  Helper wrapper for calling an individual plotting script from Python.

- `analysis/generate_tables.py`
  Converts CSV summaries into Markdown tables.

### `configs/`

YAML experiment configuration files. These are the main experiment definitions for the Linux/AWS pipeline.

- `configs/baseline_2d.yaml`
  Standard 2D nnU-Net baseline configuration.

- `configs/baseline_3d.yaml`
  Standard 3D full-resolution nnU-Net baseline configuration.

- `configs/naive_25d_3slice.yaml`
  Naive 2.5D configuration using 3 adjacent slices.

- `configs/naive_25d_5slice.yaml`
  Naive 2.5D configuration using 5 adjacent slices.

- `configs/spacing_aware_25d.yaml`
  Spacing-aware 2.5D configuration using the BHSD spacing summary CSV.

### `docs/`

Project documentation and simple web-facing materials.

- `docs/AWS_DEPLOYMENT.md`
  Linux/AWS G5 deployment instructions for the reproducible pipeline.

- `docs/index.html`
  Minimal static homepage for repository/project presentation.

- `docs/FILE_GUIDE.md`
  This file. Repository file-by-file guide.

### `evaluation/`

Evaluation code for segmentation metrics, fold aggregation, and statistical testing.

- `evaluation/__init__.py`
  Marks `evaluation` as a Python package.

- `evaluation/README.md`
  Short description of the evaluation module.

- `evaluation/aggregate_results.py`
  Aggregates case-level metric CSVs into summary statistics such as mean and standard deviation.

- `evaluation/metrics.py`
  Core metric functions. Computes Dice, IoU, precision, recall, and optional Hausdorff distance.

- `evaluation/run_evaluation.py`
  Reads predicted and ground-truth `nii.gz` segmentations and writes per-case, per-class metric CSV files.

- `evaluation/statistical_tests.py`
  Performs paired Wilcoxon signed-rank tests and paired t-tests between models.

### `figures/`

Plotting scripts for publication-style visual outputs.

- `figures/README.md`
  Short description of the figure module.

- `figures/class_comparison.py`
  Generates per-class comparison bar charts across models and saves PNG, SVG, and summary CSV outputs.

- `figures/dsc_boxplot.py`
  Generates boxplots for metrics such as Dice.

- `figures/learning_curve.py`
  Generates learning curve plots from CSV logs containing columns such as `epoch`, `train_loss`, and `val_loss`.

### `nnunet25d/`

Custom 2.5D nnU-Net v2 extensions used in this project.

- `nnunet25d/__init__.py`
  Marks `nnunet25d` as a Python package.

- `nnunet25d/README.md`
  Overview of the 2.5D extension package and its supported trainers.

- `nnunet25d/dataloader_25d.py`
  Base custom dataloader for naive 2.5D input construction. Stacks adjacent slices while keeping only the center slice as the target.

- `nnunet25d/dataloader_spacing_aware.py`
  Spacing-aware variant of the 2.5D dataloader. Uses `bhsd_spacing_summary.csv` to adapt the z-slice step.

- `nnunet25d/install_extension.py`
  Installs the custom 2.5D package and copies the trainer/dataloader integration files into the active nnU-Net environment.

- `nnunet25d/optional_transformer.py`
  Placeholder file for future transformer-based or slice-attention research extensions. Not implemented yet.

- `nnunet25d/trainer_25d.py`
  Main custom trainer definitions:
  - `nnUNetTrainer_25D`
  - `nnUNetTrainer_25D_5Slice`
  - `nnUNetTrainer_SpacingAware25D`

- `nnunet25d/trainer_25d_5slice.py`
  Thin export wrapper for the 5-slice trainer.

- `nnunet25d/trainer_spacing_aware.py`
  Thin export wrapper for the spacing-aware trainer.

### `nnUNet_data/`

Local dataset, preprocessed nnU-Net files, and training results. This directory is intentionally excluded from Git tracking and is treated as runtime data, not repository source code.

Typical contents:
- `nnUNet_raw/`
- `nnUNet_preprocessed/`
- `nnUNet_results/`

### `results/`

Generated research outputs only. This directory is intended for machine-generated CSVs, summaries, plots, and tables.

- `results/README.md`
  Explains the intended purpose of the results directory.

### `scripts/`

Main CLI entrypoints for running the reproducible pipeline.

- `scripts/README.md`
  Short overview of the script directory.

- `scripts/analyze.sh`
  Runs the report-building stage from a combined case-metrics CSV.

- `scripts/collect_results.sh`
  Merges experiment-level case-metrics CSV files into one combined CSV in `results/aggregated/`.

- `scripts/evaluate.sh`
  Linux bash entrypoint for evaluation using a config name.

- `scripts/infer.sh`
  Linux bash entrypoint for inference using a config name.

- `scripts/install_extension.py`
  Thin wrapper that calls `nnunet25d/install_extension.py`.

- `scripts/nnunet_2d_baseline.ps1`
  Legacy Windows PowerShell baseline script for local 2D training.

- `scripts/nnunet_3d_baseline.ps1`
  Legacy Windows PowerShell baseline script for local 3D training.

- `scripts/nnunet_25d_baseline.ps1`
  Legacy Windows PowerShell baseline script for local 2.5D training.

- `scripts/preprocess.sh`
  Linux bash entrypoint for nnU-Net planning and preprocessing using a config name.

- `scripts/prepare_inference_data.py`
  Builds fold-specific validation inference directories from `imagesTr/labelsTr` using `splits_final.json`.

- `scripts/run_all.sh`
  Runs the predefined end-to-end baseline experiment configs in sequence. Custom 2.5D configs are intentionally excluded because they currently support training only.

- `scripts/run_experiment.py`
  Central Python experiment runner. Handles preprocess, train, infer, evaluate, and run-all stages from a YAML config.

- `scripts/run_experiment.sh`
  Linux bash entrypoint that calls `run_experiment.py run_all` for supported configs. Custom 2.5D configs must use `python scripts/run_experiment.py train --config ...` instead.

- `scripts/setup_env.sh`
  Creates or updates the conda environment from `environment.yml`.

- `scripts/train.sh`
  Linux bash entrypoint for training using a config name.

## Top-Level Files

### `.gitignore`

Defines untracked/generated content. Important entries:
- excludes `nnUNet_data/`
- excludes Python cache files
- excludes generated contents under `results/`
- preserves `results/README.md`

### `bhsd_spacing_summary.csv`

CSV summary of BHSD volume size and spacing. Used by the spacing-aware 2.5D dataloader and useful for dataset analysis.

### `environment.yml`

Main conda environment specification for the Linux/AWS pipeline.

### `README.md`

Repository-level overview, goals, layout, and quick-start instructions.

### `requirements.txt`

Lightweight pip-style dependency list.

### `SMOKE_TEST.md`

Records the previously completed local smoke tests for:
- 2D baseline
- 3D baseline
- 2.5D baseline

## Notes on Current State

### What is already formalized

- Linux/AWS-compatible structure
- config-driven experiment entrypoints
- baseline and 2.5D trainer definitions
- evaluation and report-generation scaffolding

### What is still a work in progress

- Full end-to-end server validation for all experiment variants
- Dedicated post-training inference/evaluation support for the custom 2.5D trainers
- Final spacing-aware 2.5D research logic
- Future transformer/anisotropic extensions
- Possible cleanup or rewrite of `.github/copilot-instructions.md`

### Legacy vs current entrypoints

Prefer these for ongoing work:
- `configs/*.yaml`
- `scripts/*.sh`
- `scripts/run_experiment.py`
- `nnunet25d/`

Treat these as local-history references:
- `scripts/*.ps1`

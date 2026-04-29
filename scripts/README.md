# Script Entry Points

This directory contains the production entry points for Linux and AWS execution, plus a few legacy Windows launchers.

## Bash scripts

- `setup_env.sh`: create or update the conda environment
- `preprocess.sh`: run nnU-Net planning and preprocessing from a config
- `train.sh`: run model training from a config
- `infer.sh`: run inference from a config
- `evaluate.sh`: aggregate metrics from a config
- `run_experiment.sh`: execute preprocess, train, infer, and evaluate in sequence for supported configs
- `run_all.sh`: run the predefined end-to-end baseline experiments
- `collect_results.sh`: merge per-experiment case metrics into one CSV
- `analyze.sh`: generate summary tables, plots, and pairwise statistical tests

## Python scripts

- `run_experiment.py`: central config-driven Python CLI
- `install_extension.py`: install custom 2.5D trainers into the active nnU-Net environment
- `prepare_inference_data.py`: stage fold-specific validation inputs and labels for inference/evaluation

## 2.5D naming

- legacy custom 2.5D baseline: `baseline_25d_3slide`
- new attention-based models: `csam_bottleneck_3slide`, `csam_3slide`, `csam_5slide`

## Windows helper scripts

- `package_aws_bundle.ps1`: build a local AWS upload bundle; default is code-only, add `-IncludeData` only when needed
- `nnunet_2d_baseline.ps1`: legacy local 2D baseline launcher
- `nnunet_3d_baseline.ps1`: legacy local 3D baseline launcher
- `nnunet_25d_baseline.ps1`: legacy local 3-slide baseline launcher

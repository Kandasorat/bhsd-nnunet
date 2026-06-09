# 2.5D Server Run Guide

This guide is for running the BHSD custom 2.5D experiments on a Linux server such as a Google Cloud VM.

## Scope

The repository now keeps the custom 2.5D code in three internal groups:

- `nnunet25d/baseline/`
  Legacy 3-slide channel-stacking baseline trainers.
- `nnunet25d/csam/`
  New CSAM feature-fusion models and trainers.
- `nnunet25d/common/`
  Shared 2.5D dataloaders.

Top-level files under `nnunet25d/` are still preserved as compatibility shims, so existing training commands do not need to change.

## Recommended Config Names

- `baseline_25d_3slide`
  Legacy simple 3-slide channel-stacking baseline.
- `csam_bottleneck_3slide`
  New bottleneck-only CSAM fusion.
- `csam_3slide`
  New multi-scale CSAM fusion with `K=3`.
- `csam_5slide`
  Optional multi-scale CSAM fusion with `K=5`.

## One-Time Environment Check

From the server shell:

```bash
cd ~/project/repo
source ~/.bashrc
conda activate bhsd-nnunet

git status
git log --oneline -5
nvidia-smi
tmux ls
```

You should also confirm your nnU-Net data roots:

```bash
echo "$nnUNet_raw"
echo "$nnUNet_preprocessed"
echo "$nnUNet_results"
```

Expected Linux-style paths usually look like:

- `/home/<user>/project/data/nnUNet_raw`
- `/home/<user>/project/data/nnUNet_preprocessed`
- `/home/<user>/project/outputs/nnUNet_results`

## Refresh The Repo Before Running

Always pull the newest code before starting a fresh server run:

```bash
cd ~/project/repo
git pull origin main
```

This is especially important now that the `nnunet25d` package has been reorganized internally into `baseline`, `csam`, and `common`.

## Reinstall The Custom 2.5D Extension

After every pull that changes `nnunet25d/`, reinstall the extension into the active environment:

```bash
cd ~/project/repo
source ~/.bashrc
conda activate bhsd-nnunet
python scripts/install_extension.py
```

This copies:

- the helper package `nnunet25d`
- the legacy 2.5D trainer shims
- the CSAM trainer shim

into the active nnU-Net environment.

## Smoke Check Before Training

If you changed the custom 2.5D code, run the smoke test first:

```bash
cd ~/project/repo
source ~/.bashrc
conda activate bhsd-nnunet
python smoke_test_25d_feature_fusion.py
```

This checks:

- trainer import
- network forward shapes
- backward pass
- CUDA path when available

## Start Training In tmux

Recommended tmux session names:

- `b25d3`
- `csam3`
- `csam5`
- `csam_bottleneck`

Example for `csam_3slide`:

```bash
tmux new -s csam3
cd ~/project/repo
source ~/.bashrc
conda activate bhsd-nnunet
git pull origin main
python scripts/install_extension.py
python scripts/run_experiment.py train --config csam_3slide
```

Example for the old baseline:

```bash
tmux new -s b25d3
cd ~/project/repo
source ~/.bashrc
conda activate bhsd-nnunet
git pull origin main
python scripts/install_extension.py
python scripts/run_experiment.py train --config baseline_25d_3slide
```

Example for bottleneck-only CSAM:

```bash
tmux new -s csam_bottleneck
cd ~/project/repo
source ~/.bashrc
conda activate bhsd-nnunet
git pull origin main
python scripts/install_extension.py
python scripts/run_experiment.py train --config csam_bottleneck_3slide
```

Example for optional `K=5` CSAM:

```bash
tmux new -s csam5
cd ~/project/repo
source ~/.bashrc
conda activate bhsd-nnunet
git pull origin main
python scripts/install_extension.py
python scripts/run_experiment.py train --config csam_5slide
```

Useful tmux commands:

- detach: `Ctrl+b`, then `d`
- reattach: `tmux attach -t csam3`
- list sessions: `tmux ls`

## Direct nnU-Net Trainer Commands

If you want to bypass `run_experiment.py`, the trainer names are:

- `nnUNetTrainer_25D`
- `nnUNetTrainer25DCSAMBottleneck`
- `nnUNetTrainer25DCSAM`
- `nnUNetTrainer25DCSAM_5Slide`

Examples:

```bash
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer_25D
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAMBottleneck
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM_5Slide
```

## Validate A Finished Run

For final validation of a completed fold:

```bash
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM --val
```

Equivalent examples:

```bash
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer_25D --val
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAMBottleneck --val
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM_5Slide --val
```

## Debug Fallback With `checkpoint_latest.pth`

nnU-Net validation expects `checkpoint_final.pth`. If training stopped early and you only want a temporary debug validation:

```bash
cp \
  "$nnUNet_results/Dataset001_BHSD/nnUNetTrainer25DCSAM__nnUNetPlans__2d/fold_0/checkpoint_latest.pth" \
  "$nnUNet_results/Dataset001_BHSD/nnUNetTrainer25DCSAM__nnUNetPlans__2d/fold_0/checkpoint_final.pth"

nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM --val
```

Treat that result as debug-only, not final.

## Output Folders

These experiments write to separate nnU-Net result directories and do not overwrite:

- standard 2D baseline
- standard 3D full-resolution baseline
- legacy 3-slide 2.5D baseline
- CSAM bottleneck runs
- CSAM multi-scale runs

Expected folder names under `$nnUNet_results/Dataset001_BHSD/`:

- `nnUNetTrainer_25D__nnUNetPlans__2d`
- `nnUNetTrainer25DCSAMBottleneck__nnUNetPlans__2d`
- `nnUNetTrainer25DCSAM__nnUNetPlans__2d`
- `nnUNetTrainer25DCSAM_5Slide__nnUNetPlans__2d`

## Quick Failure Checklist

If `csam_3slide` fails at startup:

1. Run `git pull origin main`.
2. Run `python scripts/install_extension.py` again.
3. Confirm the active environment is `bhsd-nnunet`.
4. Re-run `python smoke_test_25d_feature_fusion.py`.
5. Confirm the server has the latest reorganized `nnunet25d` package, not an older cached copy.

## Notes

- `csam_3slide` currently maps to the multi-scale CSAM trainer.
- `csam_bottleneck_3slide` is the cleaner ablation against the legacy 3-slide baseline.
- The old baseline is still kept intentionally and has not been removed.

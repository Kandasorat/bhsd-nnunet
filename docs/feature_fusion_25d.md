# 2.5D Feature Fusion Models

## Overview

This project now keeps three distinct 2D-derived slice-context variants:

1. `legacy 2.5D 3-slide baseline`
   Input slices such as `[z-1, z, z+1]` are stacked as channels and passed into a standard 2D nnU-Net.
2. `CSAM bottleneck fusion`
   Each slice is encoded independently by a shared 2D encoder and only the bottleneck features are fused.
3. `CSAM multi-scale fusion`
   Each encoder scale is reshaped to `[B, K, C_s, H_s, W_s]` and fused with a scale-specific center-guided attention block.

The standard 2D and 3D full-resolution baselines are unchanged.

## Shapes

Input:

- `[B, K, H, W]`
- `[B, K, 1, H, W]`
- trainer-time stacked format remains compatible with the existing 2.5D dataloader

Output:

- raw logits for the center slice only
- first output shape: `[B, num_classes, H, W]`
- when deep supervision is enabled, additional lower-resolution outputs are returned in standard nnU-Net order

## Trainers

- `nnUNetTrainer25DCSAMBottleneck`
  Separate CSAM bottleneck-only fusion trainer.
- `nnUNetTrainer25DCSAM`
  Primary CSAM multi-scale fusion trainer for `K=3`.
- `nnUNetTrainer25DCSAM_5Slide`
  Optional CSAM multi-scale fusion trainer for `K=5`.
- `nnUNetTrainer25DFeatureFusion`
  Backward-compatible alias for the original bottleneck-only feature-fusion trainer.

## Smoke Test

Run from the project root:

```powershell
$env:PYTHONPATH="C:\Users\92127\OneDrive - UNSW\project_linpeng\code"
& "C:\Users\92127\anaconda3\envs\bhsd\python.exe" "C:\Users\92127\OneDrive - UNSW\project_linpeng\code\smoke_test_25d_feature_fusion.py"
```

The smoke test checks:

- bottleneck fusion forward for `K=3`
- multi-scale fusion forward for `K=3`
- optional multi-scale fusion forward for `K=5`
- backward pass on CPU
- backward pass on CUDA when available
- parameter counts
- trainer import and nnU-Net trainer lookup

## Training

Recommended configs:

- `configs/baseline_25d_3slide.yaml`
- `configs/csam_bottleneck_3slide.yaml`
- `configs/csam_3slide.yaml`
- `configs/csam_5slide.yaml`

Project launcher examples:

```powershell
python scripts/run_experiment.py train --config baseline_25d_3slide
python scripts/run_experiment.py train --config csam_bottleneck_3slide
python scripts/run_experiment.py train --config csam_3slide
python scripts/run_experiment.py train --config csam_5slide
```

Direct Linux server and `tmux` commands are documented in [run_25d_feature_fusion_server.md](/C:/Users/92127/OneDrive%20-%20UNSW/project_linpeng/code/docs/run_25d_feature_fusion_server.md).

## Output Folders

nnU-Net keeps each trainer in a separate result directory, so the feature-fusion runs do not overwrite the existing 2D, 3D, or simple 2.5D baselines.

Expected folder patterns:

- `nnUNetTrainer25DCSAMBottleneck__nnUNetPlans__2d`
- `nnUNetTrainer25DCSAM__nnUNetPlans__2d`
- `nnUNetTrainer25DCSAM_5Slide__nnUNetPlans__2d`

These live under:

- `C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_results\Dataset001_BHSD\`

## Known Limitations

- No full training is run locally as part of this implementation step.
- The repository still does not provide a dedicated standalone inference pipeline for the custom 2.5D trainers through `scripts/run_experiment.py infer`.
- Attention weights are available in-memory after a forward pass, but persistent logging is not added yet.
- Binary Dice evaluation and report integration are still pending future stages.

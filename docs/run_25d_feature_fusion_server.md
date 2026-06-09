# 2.5D Feature Fusion Server Commands

## Environment

Set the nnU-Net paths before running any command:

```powershell
$env:nnUNet_raw="C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_raw"
$env:nnUNet_preprocessed="C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_preprocessed"
$env:nnUNet_results="C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_results"
$env:PYTHONPATH="C:\Users\92127\OneDrive - UNSW\project_linpeng\code"
```

Install or refresh the custom 2.5D extension:

```powershell
& "C:\Users\92127\anaconda3\envs\bhsd\python.exe" "C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnunet25d\install_extension.py"
```

Dataset and configuration:

- dataset: `Dataset001_BHSD`
- configuration: `2d`
- plans: `nnUNetPlans`

## Train

Fold-0 legacy `3-slide` baseline:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer_25D
```

Fold-0 `K=3` CSAM bottleneck fusion:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAMBottleneck
```

Fold-0 `K=3` CSAM multi-scale fusion:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM
```

Optional Fold-0 `K=5` CSAM multi-scale fusion:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM_5Slide
```

## Validate Final Checkpoint

Final validation for legacy `3-slide` baseline:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer_25D --val
```

Final validation for CSAM bottleneck fusion:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAMBottleneck --val
```

Final validation for CSAM multi-scale fusion:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM --val
```

Final validation for optional `K=5` CSAM multi-scale fusion:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM_5Slide --val
```

## Debug-Only Fallback Using `checkpoint_latest.pth`

nnU-Net validation normally expects `checkpoint_final.pth`. If training has not finished and you only want a temporary debug validation, copy `checkpoint_latest.pth` to `checkpoint_final.pth`, run validation, and treat the result as non-final.

Debug fallback for CSAM multi-scale fusion:

```powershell
Copy-Item `
  "C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_results\Dataset001_BHSD\nnUNetTrainer25DCSAM__nnUNetPlans__2d\fold_0\checkpoint_latest.pth" `
  "C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_results\Dataset001_BHSD\nnUNetTrainer25DCSAM__nnUNetPlans__2d\fold_0\checkpoint_final.pth" `
  -Force
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer25DCSAM --val
```

Use the same pattern for the bottleneck and `K=5` trainers if needed.

## Output Folders

These trainers write to distinct output roots, so they will not overwrite:

- existing 2D baseline
- existing 3D full-resolution baseline
- existing simple 2.5D stacking baseline
- bottleneck feature fusion runs
- multi-scale feature fusion runs

Expected roots:

- `...\Dataset001_BHSD\nnUNetTrainer_25D__nnUNetPlans__2d`
- `...\Dataset001_BHSD\nnUNetTrainer25DCSAMBottleneck__nnUNetPlans__2d`
- `...\Dataset001_BHSD\nnUNetTrainer25DCSAM__nnUNetPlans__2d`
- `...\Dataset001_BHSD\nnUNetTrainer25DCSAM_5Slide__nnUNetPlans__2d`

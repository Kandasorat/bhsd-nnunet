# BHSD Binary Retraining

This document adds binary hemorrhage retraining while preserving the existing five-class BHSD workflow.

## Task Separation

Five-class segmentation uses:

- dataset: `Dataset001_BHSD`
- labels: `0 background`, `1 epidural`, `2 intraparenchymal`, `3 intraventricular`, `4 subarachnoid`, `5 subdural`
- headline metric: strict five-class foreground mean Dice

Binary hemorrhage segmentation uses:

- dataset: `Dataset002_BHSD_Binary`
- labels: `0 background`, `1 hemorrhage`
- headline metric: binary hemorrhage Dice

Warning:
Binary hemorrhage Dice is not directly comparable with strict five-class Dice because binary training merges EDH, IPH, IVH, SAH and SDH into one foreground class.

## 1. Create the Binary Raw Dataset

Source dataset stays unchanged:

- `nnUNet_data/nnUNet_raw/Dataset001_BHSD`

Create the binary raw dataset:

```powershell
cd "C:\Users\92127\OneDrive - UNSW\project_linpeng\code"
& "C:\Users\92127\anaconda3\envs\bhsd\python.exe" scripts/create_bhsd_binary_dataset.py `
  --src-dataset "C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_raw\Dataset001_BHSD" `
  --dst-dataset "C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_raw\Dataset002_BHSD_Binary"
```

Expected output CSV:

- `outputs/bhsd_binary_dataset_check.csv`

## 2. Sanity Check Result

The conversion script prints:

- number of training cases
- number of binary-positive cases
- total binary foreground voxels
- min/median/max foreground voxels per positive case

The binary labels must contain only `{0,1}`.

## 3. Plan and Preprocess

Dataset ID `2` is used because the raw folder is `Dataset002_BHSD_Binary`.

```bash
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
```

## 4. Fold-0 Training Commands

Standard 2D binary:

```bash
python scripts/run_experiment.py train --config baseline_2d_binary
```

Simple 2.5D binary, 3 slices:

```bash
python scripts/run_experiment.py train --config baseline_25d_3slide_binary
```

Optional simple 2.5D binary, 5 slices:

```bash
python scripts/run_experiment.py train --config baseline_25d_5slide_binary
```

Standard 3D full-resolution binary:

```bash
python scripts/run_experiment.py train --config baseline_3d_binary
```

CSAM binary, 3 slices:
Only use after the binary CSAM verification below passes.

```bash
python scripts/run_experiment.py train --config csam_3slide_binary
```

## 5. Binary CSAM Verification

Verify imports and binary output shape first:

```bash
python scripts/verify_csam_binary.py
```

This verification expects `Dataset002_BHSD_Binary` to already exist and to have completed `nnUNetv2_plan_and_preprocess -d 2`.

Checks covered:

- direct import of `nnUNetTrainer25DCSAM`
- shim import through `nnunetv2`
- binary `num_segmentation_heads == 2`
- binary CSAM highest-resolution output shape `[2, 2, 256, 256]`
- shared-encoder 2.5D input contract remains unchanged

## 6. Validation Commands

Standard 2D binary final validation:

```bash
nnUNetv2_train Dataset002_BHSD_Binary 2d 0 -tr nnUNetTrainer_BHSDEarlyStop --val
```

Standard 3D binary final validation:

```bash
nnUNetv2_train Dataset002_BHSD_Binary 3d_fullres 0 -tr nnUNetTrainer_BHSDEarlyStop --val
```

Simple 2.5D binary final validation:

```bash
nnUNetv2_train Dataset002_BHSD_Binary 2d 0 -tr nnUNetTrainer_25D --val
```

CSAM binary final validation:

```bash
nnUNetv2_train Dataset002_BHSD_Binary 2d 0 -tr nnUNetTrainer25DCSAM --val
```

Debug-only fallback if only `checkpoint_latest.pth` exists:

```bash
cp \
  "$nnUNet_results/Dataset002_BHSD_Binary/nnUNetTrainer25DCSAM__nnUNetPlans__2d/fold_0/checkpoint_latest.pth" \
  "$nnUNet_results/Dataset002_BHSD_Binary/nnUNetTrainer25DCSAM__nnUNetPlans__2d/fold_0/checkpoint_final.pth"
```

Treat that as debug-only, not final.

## 7. 2.5D / CSAM Validation Compatibility

The current trainer-driven formal validation path reconstructs adjacent-slice input through `_stack_case_for_inference()`.

That means for each axial slice `z` it reconstructs neighboring slices with boundary clamping and writes the center-slice prediction back into the full volume.

This is currently compatible with:

- simple 2.5D validation
- CSAM 2.5D validation

Standalone `nnUNetv2_predict` inference is still not implemented for the custom 2.5D trainers in `scripts/run_experiment.py`.

## 8. Binary Evaluation

Binary-trained models should use the dedicated binary evaluation script:

```bash
python scripts/evaluate_binary_segmentation.py \
  --gt-dir /path/to/ground_truth_binary_labels \
  --pred-dir /path/to/predictions \
  --model-name csam_3slide_binary \
  --out-dir /path/to/output_dir
```

Outputs:

- `binary_segmentation_per_case.csv`
- `binary_segmentation_summary.csv`

The evaluator enforces exact case-ID matching between `--gt-dir` and `--pred-dir`, rejects labels outside `{0,1}`, and reports Hausdorff distance when the dependency is available and both masks are non-empty.

## 9. Difference From Merged Binary Evaluation of Five-Class Models

This binary retraining workflow is different from taking five-class model outputs and post-hoc merging them into one foreground label.

Here the models are retrained from scratch on:

- dataset: `Dataset002_BHSD_Binary`
- labels: `0 background`, `1 hemorrhage`

So the resulting Dice is a training-task-specific binary hemorrhage Dice, not a relabeled five-class Dice.

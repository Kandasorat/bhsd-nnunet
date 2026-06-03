# Smoke Test Summary

Date: 2026-05-01

This document records short smoke tests for the BHSD nnU-Net baselines on the local Windows machine.

## Goal

The purpose of these tests was only to verify that each training pipeline can start correctly.

Each run was stopped early after confirming:

- nnU-Net entered `Epoch 0`
- GPU utilization increased as expected
- no immediate runtime error occurred

These were not full training runs.

## Environment

- OS: Windows
- Conda environment: `bhsd`
- Dataset: `Dataset001_BHSD`
- Data paths:
  - `nnUNet_raw=C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_raw`
  - `nnUNet_preprocessed=C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_preprocessed`
  - `nnUNet_results=C:\Users\92127\OneDrive - UNSW\project_linpeng\code\nnUNet_data\nnUNet_results`
- Additional setting used for stability on this machine:
  - `nnUNet_n_proc_DA=0`

## 2D Baseline

Command:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 --npz --disable_checkpointing
```

Result:

- Training started successfully
- Reached `Epoch 0`
- Learning rate reported as `0.01`
- No immediate runtime error

Latest log:

- [nnUNet_data/nnUNet_results/Dataset001_BHSD/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_2026_5_1_10_26_07.txt](</C:/Users/92127/OneDrive - UNSW/project_linpeng/code/nnUNet_data/nnUNet_results/Dataset001_BHSD/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_2026_5_1_10_26_07.txt>)

Observed GPU usage:

- utilization rose from about `21%` to `100%`
- memory usage rose from about `1.1 GB` to about `5.7 GB`

Notes:

- `hiddenlayer` was missing for architecture plotting
- this did not affect training startup

## 3D Full-Resolution Baseline

Command:

```powershell
nnUNetv2_train Dataset001_BHSD 3d_fullres 0 --npz --disable_checkpointing
```

Result:

- Training started successfully
- Reached `Epoch 0`
- Learning rate reported as `0.01`
- No immediate runtime error

Latest log:

- [nnUNet_data/nnUNet_results/Dataset001_BHSD/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2026_5_1_10_27_17.txt](</C:/Users/92127/OneDrive - UNSW/project_linpeng/code/nnUNet_data/nnUNet_results/Dataset001_BHSD/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2026_5_1_10_27_17.txt>)

Observed GPU usage:

- utilization rose from about `18%` to `100%`
- memory usage rose from about `0.8 GB` to about `5.7 GB`

Notes:

- `hiddenlayer` was missing for architecture plotting
- this did not affect training startup

## 2.5D Baseline

Command:

```powershell
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer_25D --npz --disable_checkpointing
```

Result:

- Training started successfully
- Reached `Epoch 0`
- Learning rate reported as `0.01`
- No immediate runtime error

Latest log:

- [nnUNet_data/nnUNet_results/Dataset001_BHSD/nnUNetTrainer_25D__nnUNetPlans__2d/fold_0/training_log_2026_5_1_10_29_36.txt](</C:/Users/92127/OneDrive - UNSW/project_linpeng/code/nnUNet_data/nnUNet_results/Dataset001_BHSD/nnUNetTrainer_25D__nnUNetPlans__2d/fold_0/training_log_2026_5_1_10_29_36.txt>)

Observed GPU usage:

- utilization rose from about `15%` to `100%`
- memory usage rose from about `0.6 GB` to about `5.6 GB`

Notes:

- `hiddenlayer` was missing for architecture plotting
- this did not affect training startup
- this smoke test used the installed custom trainer `nnUNetTrainer_25D`

## Overall Conclusion

All three pipelines passed startup-level smoke testing on the local machine:

- 2D baseline
- 3D full-resolution baseline
- 2.5D custom trainer baseline

The smoke tests confirm that the training entry points, dataset paths, GPU execution, and trainer initialization are working at startup level.

They do not yet confirm:

- full epoch completion
- final validation
- inference/export behavior
- long-run stability
- training loss trends across full epochs

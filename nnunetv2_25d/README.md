# nnU-Net 2.5D Extension

This folder contains a minimal 2.5D extension for nnU-Net v2.

## What it does

- Reuses the standard `2d` nnU-Net configuration.
- Reuses the same preprocessed dataset, for example `Dataset001_BHSD`.
- Replaces single-slice loading with stacked adjacent slices:
  - `nnUNetTrainer_25D`: `z-1, z, z+1`
  - `nnUNetTrainer_25D_5Slice`: `z-2, z-1, z, z+1, z+2`
- Uses only the center slice as the target.

## Boundary handling

- First slice: duplicate the first slice as needed
- Last slice: duplicate the last slice as needed

## Install

Run:

```bash
python nnunetv2_25d/install_25d_extension.py
```

## Train

Three-slice baseline:

```bash
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer_25D
```

Five-slice baseline:

```bash
nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer_25D_5Slice
```

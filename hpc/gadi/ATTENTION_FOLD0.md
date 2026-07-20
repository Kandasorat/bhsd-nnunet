# BHSD fold-0 attention pilots

Two distinct paper-based experiments are provided. They must not be described
as the same 3-slice model.

## 1. Volume-wise CSAM

- Trainer: `nnUNetTrainerCSAMVolumeOfficial`
- Config: `configs/csam_official_volume32_fold0.yaml`
- PBS: `hpc/gadi/train_csam_volume_fold0.pbs`
- Input: ordered overlapping windows of 32 slices from one volume
- Output: segmentation for all 32 slices
- Full validation: overlapping sequence and 256x256 spatial windows cover the
  complete validation volume

Submit from the repository root:

```bash
cd /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet
qsub hpc/gadi/train_csam_volume_fold0.pbs
```

## 2. Official CSA-Net three-slice model

- Trainer: `nnUNetTrainer25DCSANetOfficial`
- Config: `configs/csa_net_official_3slice_fold0.yaml`
- PBS: `hpc/gadi/train_csa_net_fold0.pbs`
- Input: previous, center, and next slice
- Output: center-slice six-class segmentation
- Architecture: official R50-ViT-B/16 CSA-Net with registered 16-head
  cross-slice attention

The official ImageNet-21k pretrained file is required at:

```text
/scratch/ke17/bhsd-nnunet/software/pretrained/R50+ViT-B_16.npz
```

The file is linked by the official repository README. Upload it once; it must
not be committed to Git.

Submit:

```bash
cd /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet
qsub hpc/gadi/train_csa_net_fold0.pbs
```

## Shared experiment rules

Both pilots use:

- Dataset001_BHSD and the existing fold 0 split;
- 256x256 spatial patches;
- maximum 1000 epochs;
- early stopping on `ema_fg_dice`, minimum 300 epochs, then patience 100, minimum delta 0.0001;
- `checkpoint_best.pth` for full validation;
- saved validation probabilities (`--npz`).

Run them separately first because each requests one V100. Do not submit either
as a five-fold array until fold 0 completes, memory use is checked, and its
per-class validation results are reviewed.

# BHSD binary 2D and 3D baselines on Gadi

This is a separate task from the six-label BHSD experiment. The raw label maps
remain unchanged, while nnU-Net trains one region target:

```text
hemorrhage = {EDH, IPH, IVH, SAH, SDH} = {1, 2, 3, 4, 5}
```

The binary Dice is therefore not directly comparable with the multiclass macro
Dice.

## Shared controls

- Dataset: `Dataset002_BHSD_Binary`
- Exact same five case splits as `Dataset001_BHSD`
- Trainer: `nnUNetTrainer_BHSDEarlyStop`
- Maximum epochs: 1000
- Early stopping: `ema_fg_dice`, minimum 300 epochs, then patience 100, minimum delta 0.0001
- Formal validation and inference: `checkpoint_best.pth`
- 2D patch: 256x256 (trainer override)
- 3D patch: nnU-Net planned depth with 256x256 in-plane size

## 1. Prepare Dataset002

After pulling the current `main`, submit from the log directory:

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
cd "$BHSD_ROOT/logs"
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/prepare_binary_dataset.pbs"
```

The preparation job creates and verifies the region dataset, runs nnU-Net
planning/preprocessing, and copies the exact Dataset001 five-fold split. Do not
submit training until this job finishes with exit status 0.

## 2. Train the five 2D folds

```bash
cd "$BHSD_ROOT/logs"
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_2d_binary_folds.pbs"
```

## 3. Train the five 3D folds

Submit after binary 2D has started successfully (or after it finishes if GPU
queue pressure should be minimized):

```bash
cd "$BHSD_ROOT/logs"
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_3d_binary_folds.pbs"
```

Monitor with `qstat -u "$USER"`. Each array contains folds 0 through 4 and is
rerunnable; existing checkpoints cause the corresponding fold to resume.

## 4. Binary 2.5D and source-faithful attention fold-0 pilots

These are additional fold-0 experiments, not reruns of the completed binary
2D/3D baselines. They use the same Dataset002 fold split and separate result
namespaces:

```bash
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_binary_25d_3slice_fold0.pbs"
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_csam_source_faithful_binary_fold0.pbs"
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_csa_net_source_faithful_binary_fold0.pbs"
```

Run the four-way source-faithful smoke array first and require all subjobs to
finish with exit status 0. See `IMMEDIATE_NEXT_RUNBOOK.md` for the enforced
submission order.

# BHSD standard three-slice 2.5D fold 0

This pilot is the simple 2.5D baseline, not CSAM and not CSA-Net.

- Dataset: multiclass `Dataset001_BHSD`
- Fold: 0, using the existing Dataset001 five-fold split
- Input for centre slice z: slices z-1, z, and z+1 as three channels
- Boundary rule: clamp to the first or last available slice
- Target: centre-slice six-label segmentation
- Spatial patch: 256x256
- Trainer/result namespace: `nnUNetTrainer_25D_HarmonizedMin300Patience100`
- Maximum epochs: 1000
- Early stopping: `ema_fg_dice`, minimum 300 epochs, then patience 100, minimum delta 0.0001
- Formal validation: `checkpoint_best.pth`
- Validation probabilities: saved for consistent baseline comparison

Submit after pulling the current `main`:

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
cd "$BHSD_ROOT/logs"
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_25d_3slice_fold0.pbs"
```

The trainer name deliberately creates a clean result path distinct from the
older `nnUNetTrainer_25D` fold-0 run that used a different stopping policy.
The script is rerunnable and resumes fold 0 only when a checkpoint is present
inside this new harmonized result path.

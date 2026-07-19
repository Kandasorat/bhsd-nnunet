# Gadi training jobs

These PBS array jobs run the five formal cross-validation folds for the BHSD
2D and 3D nnU-Net baselines. Experiment settings remain in `configs/`; the PBS
files only declare Gadi resources and select the array fold.

Both jobs use the shared early-stopping trainer and request full-case validation
with `checkpoint_best.pth`. Model outputs are written below
`/scratch/ke17/bhsd-nnunet/runs/nnUNet_results`, while run metadata is written
below `/scratch/ke17/bhsd-nnunet/runs/experiment_metadata`.

## Before submission

Update and verify the server checkout:

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
cd "$BHSD_ROOT/software/bhsd-nnunet"
git pull --ff-only origin main
git status --short
```

The status output must be empty. Submit from the log directory so PBS standard
output and error files do not appear in the Git checkout:

```bash
cd "$BHSD_ROOT/logs"
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_2d_folds.pbs"
```

After the 2D jobs have been checked, submit the 3D folds:

```bash
cd "$BHSD_ROOT/logs"
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_3d_folds.pbs"
```

Monitor all subjobs with:

```bash
qstat -u "$USER"
```

If a fold reaches its walltime before training finishes, resume the same array
index from the latest nnU-Net checkpoint. Replace `N` with the fold number:

```bash
cd "$BHSD_ROOT/logs"
qsub -r y -J N-N -v BHSD_RESUME=1 \
  "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_2d_folds.pbs"
```

Use the corresponding 3D PBS path when resuming a 3D fold. Do not use resume for
a fold that completed successfully.

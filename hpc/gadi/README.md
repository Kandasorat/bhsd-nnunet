# Production jobs for NCI Gadi

These PBS scripts are the supported Gadi launchers. Run Git, file inspection,
and `qsub` on a login node; training runs only inside PBS GPU jobs.

For a concise beginner-oriented Chinese guide covering login, Git, PBS,
monitoring, logs, and downloads, see
[`docs/GADI_PLATFORM_GUIDE_ZH.md`](../../docs/GADI_PLATFORM_GUIDE_ZH.md).

## 1. Refresh and validate the checkout

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
export nnUNet_raw="$BHSD_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$BHSD_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$BHSD_ROOT/runs/nnUNet_results"

cd "$BHSD_ROOT/software/bhsd-nnunet"
git pull --ff-only origin main
git status --short

module purge
module load python3/3.10.4
source "$BHSD_ROOT/envs/bhsd-nnunet-py310/bin/activate"
python3 scripts/check_gadi_ready.py --server
```

Do not submit from inside the Git checkout. Use the external log directory so
PBS stdout and stderr never dirty the repository:

```bash
cd "$BHSD_ROOT/logs"
```

## 2. Multiclass jobs

```bash
# Five-fold 2D and 3D baselines
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_2d_folds.pbs"
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_3d_folds.pbs"

# Standard three-slice 2.5D fold 0
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_25d_3slice_fold0.pbs"

# Harmonized multiclass A1-A8 module screen (eight fold-0 array subjobs)
# Submit after the standard A0 fold-0 job has completed.
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_25d_attention_screen_fold0.pbs"

# Tier-3 nnU-Net adaptations; first implement the source-faithful references
# described in docs/ATTENTION_REPRODUCTION_POLICY.md
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_csam_volume_fold0.pbs"
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_csa_net_fold0.pbs"
```

CSA-Net requires:

```text
/scratch/ke17/bhsd-nnunet/software/pretrained/R50+ViT-B_16.npz
```

## 3. Binary jobs

Dataset002 is a separate derived dataset; Dataset001 remains unchanged.

```bash
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/prepare_binary_dataset.pbs"
```

After preparation finishes with exit status 0:

```bash
cd "$BHSD_ROOT/software/bhsd-nnunet"
python3 scripts/check_gadi_ready.py --server --require-binary
cd "$BHSD_ROOT/logs"

qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_2d_binary_folds.pbs"
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_3d_binary_folds.pbs"
```

## 4. Monitoring and resuming

```bash
qstat -t -u "$USER"
qstat -x -t "JOB_ID[].gadi-pbs"
```

For an interrupted array fold, replace `N` and use the matching PBS file:

```bash
qsub -r y -J N-N -v BHSD_RESUME=1 \
  "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_2d_folds.pbs"
```

The scripts also detect existing checkpoints and add `--resume`. Do not remove
or overwrite a completed result directory unless that specific experiment is
being intentionally restarted.

## Shared policy

- 2D/2.5D patch remains `256 x 256`
- Dataset001 3D plan remains `28 x 256 x 256`
- maximum 1000 epochs
- minimum 300 epochs, followed by patience 100
- minimum delta 0.0001 on `ema_fg_dice`
- formal validation and inference use `checkpoint_best.pth`

Each GPU job installs the checked-out custom trainers under a `flock` lock, so
simultaneous array jobs cannot copy the extension concurrently.

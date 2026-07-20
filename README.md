# BHSD nnU-Net Research Pipeline

Reproducible BHSD intracranial-haemorrhage segmentation experiments built on
nnU-Net v2. The active execution target is NCI Gadi; datasets, preprocessing
caches, checkpoints, and predictions stay outside Git.

## Active experiments

- 2D and 3D full-resolution nnU-Net baselines
- three-slice and spacing-aware 2.5D baselines
- binary 2D, 3D, and 2.5D baselines
- upstream-architecture CSAM and CSA-Net nnU-Net adaptation pilots

The active source package is `nnunet25d/`. Historical duplicate implementations
are retained under `archive/` and are not server entrypoints.

## Fixed comparison policy

- 2D and 2.5D spatial patch: `256 x 256`
- 3D patch: keep the existing Dataset001 nnU-Net plan (`28 x 256 x 256` on Gadi)
- maximum epochs: 1000
- minimum epochs before patience monitoring: 300
- patience after epoch 300: 100
- minimum improvement: 0.0001 on `ema_fg_dice`
- formal validation and inference checkpoint: `checkpoint_best.pth`

Do not edit a patch size in place for an existing experiment. Patch-size
ablations must use a separate configuration and output directory.

## Gadi layout

```text
/scratch/ke17/bhsd-nnunet/
  software/bhsd-nnunet/       Git checkout
  envs/bhsd-nnunet-py310/     Python environment
  data/nnUNet_raw/            raw nnU-Net datasets
  data/nnUNet_preprocessed/   plans, splits, and preprocessed arrays
  runs/nnUNet_results/        checkpoints and validation predictions
  runs/experiment_metadata/   config and resource records
  logs/                       PBS stdout/stderr
```

## Update and verify on Gadi

Run these lightweight commands on a login node:

```bash
ssh ly6399@gadi.nci.org.au

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
python scripts/check_gadi_ready.py --server
```

`git status --short` should be empty. Do not train on a login node. Each PBS job
installs the current custom extension under a lock before starting GPU work.

## Submit jobs

Submit from the external log directory:

```bash
cd "$BHSD_ROOT/logs"

# Formal five-fold baselines
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_2d_folds.pbs"
qsub -r y "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_3d_folds.pbs"

# Standard three-slice 2.5D fold 0
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_25d_3slice_fold0.pbs"

# Tier-3 nnU-Net attention adaptations; run only after source-faithful references
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_csam_volume_fold0.pbs"
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_csa_net_fold0.pbs"
```

The CSA-Net job additionally requires
`$BHSD_ROOT/software/pretrained/R50+ViT-B_16.npz`.

Before any CSAM/CSA-Net submission, read
`docs/ATTENTION_REPRODUCTION_POLICY.md`. These PBS jobs adapt upstream
architectures to nnU-Net; they are not source-faithful official training
protocol reproductions.

Monitor jobs with `qstat -u "$USER"`. Closing the local terminal does not stop a
submitted non-interactive PBS job.

## Repository map

```text
configs/       experiment records
hpc/gadi/      production PBS jobs and Gadi instructions
nnunet25d/     custom trainers and paper-based modules
scripts/       config runner, extension installer, and readiness checker
evaluation/    metrics and aggregation
analysis/      tables, reports, and figures
docs/          project handoff and method notes
archive/       historical source retained for reference only
```

Start with `hpc/gadi/README.md` for execution and `docs/PROJECT_HANDOFF.md` for
the experiment history. Local `nnUNet_data/`, `results/`, and `outputs/` are
ignored by Git.

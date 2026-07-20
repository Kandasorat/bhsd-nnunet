# Running active 2.5D experiments on Gadi

The active 2.5D routes are:

- `baseline_25d_3slide`: standard previous/centre/next slice stacking
- `spacing_aware_25d`: spacing-aware three-slice sampling
- `csam_official_volume32_fold0`: paper-based volume-wise CSAM pilot
- `csa_net_official_3slice_fold0`: paper-based CSA-Net pilot

Legacy custom feature-fusion code is archived and must not be used for new
server experiments.

## Before submission

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
export nnUNet_raw="$BHSD_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$BHSD_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$BHSD_ROOT/runs/nnUNet_results"

cd "$BHSD_ROOT/software/bhsd-nnunet"
git pull --ff-only origin main

module purge
module load python3/3.10.4
source "$BHSD_ROOT/envs/bhsd-nnunet-py310/bin/activate"
python scripts/check_gadi_ready.py --server
```

## Submit

```bash
cd "$BHSD_ROOT/logs"
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_25d_3slice_fold0.pbs"
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_csam_volume_fold0.pbs"
qsub "$BHSD_ROOT/software/bhsd-nnunet/hpc/gadi/train_csa_net_fold0.pbs"
```

Run the attention pilots separately until fold 0 memory use and per-class
validation results have been reviewed. CSA-Net requires the pretrained file
documented in `hpc/gadi/ATTENTION_FOLD0.md`.

All active configs use the same maximum-1000, minimum-300, patience-100 rule
and full validation from `checkpoint_best.pth`.

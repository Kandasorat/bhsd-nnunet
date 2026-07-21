# Immediate next work: operator runbook

Run each block in order. The four completed formal 2D/3D baselines are not
submitted by this runbook.

## 1. Login and read-only preflight

```bash
ssh ly6399@gadi.nci.org.au
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
export REPO_DIR="$BHSD_ROOT/software/bhsd-nnunet"
qstat -u "$USER"
cd "$REPO_DIR"
git status --short --branch
git pull --ff-only origin main
git rev-parse HEAD
```

Stop if there are unintended jobs or uncommitted server changes. The revision
must contain or descend from `d04643e` and include the new isolated 2.5D and
`source_faithful` files.

## 2. Environment and repository readiness

```bash
export nnUNet_raw="$BHSD_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$BHSD_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$BHSD_ROOT/runs/nnUNet_results"
module purge
module load python3/3.10.4
source "$BHSD_ROOT/envs/bhsd-nnunet-py310/bin/activate"
python scripts/check_gadi_ready.py --server --require-binary
```

Confirm both new 2.5D result namespaces are clean:

```bash
test ! -e "$nnUNet_results/Dataset001_BHSD/nnUNetTrainer_25D_HarmonizedMin300Patience100__nnUNetPlans__2d/fold_0"
test ! -e "$nnUNet_results/Dataset002_BHSD_Binary/nnUNetTrainer_25D_HarmonizedMin300Patience100__nnUNetPlans__2d/fold_0"
```

## 3. Submit isolated harmonized multiclass and binary 2.5D fold 0

```bash
cd "$BHSD_ROOT/logs"
qsub "$REPO_DIR/hpc/gadi/train_25d_3slice_fold0.pbs"
qsub "$REPO_DIR/hpc/gadi/train_binary_25d_3slice_fold0.pbs"
```

## 4. Smoke-test both source-faithful BHSD ports

CSA-Net requires this file first:

```text
/scratch/ke17/bhsd-nnunet/software/pretrained/R50+ViT-B_16.npz
```

Then submit the four-job smoke array (multiclass/binary x CSAM/CSA-Net):

```bash
cd "$BHSD_ROOT/logs"
qsub "$REPO_DIR/hpc/gadi/smoke_source_faithful_attention.pbs"
```

Require all four subjobs to finish with `Exit_status = 0` and inspect:

```bash
find "$BHSD_ROOT/runs/source_faithful_smoke" -name smoke.json -print -exec cat {} \;
```

## 5. Submit source-faithful fold 0 only after both smoke tests pass

The selection rules are already frozen in the YAML configs. Do not change them
after viewing outcomes.

```bash
cd "$BHSD_ROOT/logs"
qsub "$REPO_DIR/hpc/gadi/train_csam_source_faithful_fold0.pbs"
qsub "$REPO_DIR/hpc/gadi/train_csa_net_source_faithful_fold0.pbs"
qsub "$REPO_DIR/hpc/gadi/train_csam_source_faithful_binary_fold0.pbs"
qsub "$REPO_DIR/hpc/gadi/train_csa_net_source_faithful_binary_fold0.pbs"
```

If any source-faithful run reaches the 48-hour walltime, submit the same PBS
file again. It detects `checkpoint_latest.pth` and resumes at the next epoch.

Do not submit the older `train_csam_volume_fold0.pbs` or
`train_csa_net_fold0.pbs` as source-faithful runs; they remain separately
labelled harmonized nnU-Net adaptations.

# Controlled 2.5D fusion screen (C0-C3, F1-F2)

## Purpose

The completed A1-A8 fold-0 screen identified axial slice convolution (A8) as
the best overall arm and centre-to-neighbour CSA (A5) as the best attention
arm. This follow-up tests whether their local and adaptive cross-slice signals
are complementary inside the same nnU-Net training/evaluation framework.

## Frozen arms

| ID | Trainer | Definition |
|---|---|---|
| C0 | `nnUNetTrainer_25D_Controlled` | stacked three-slice baseline |
| C1 | `nnUNetTrainer_25D_AdapterControlControlled` | shared descriptor/projection capacity control |
| C2 | `nnUNetTrainer_25D_CSACenterNeighborControlled` | controlled A5 replication |
| C3 | `nnUNetTrainer_25D_AxialSliceConvControlled` | controlled A8 replication |
| F1 | `nnUNetTrainer_25D_AxialCSASequential` | axial convolution, then CSA |
| F2 | `nnUNetTrainer_25D_AxialCSAParallel` | learnable softmax mixture of axial and CSA branches |

All adapter arms use the same 8-channel shared descriptor and zero-initialized
residual output projection. F2 starts with branch weights `[0.5, 0.5]`.

## Reproducibility boundary

All six configs fix seed 3407 in the actual `nnUNetv2_train` child process,
set `PYTHONHASHSEED`, seed Python/NumPy/PyTorch/CUDA, disable cuDNN benchmarking,
request deterministic PyTorch algorithms, and force single-thread data
augmentation. A separate data seed is reapplied after network initialization,
so modules with different parameter counts cannot shift augmentation randomness;
epoch-indexed train/validation seeds also make a resumed epoch independent of
the number of random draws made before resumption. This controls the
experiment's stochastic inputs.

PyTorch 2.7.1 reports that CUDA `nll_loss2d` and
`adaptive_avg_pool2d_backward` do not have deterministic implementations on
the tested stack. The runner therefore uses `warn_only=True`: these runs are
controlled/best-effort reproducible, not guaranteed bitwise identical across
devices or executions. Multi-seed confirmation remains necessary.

## Frozen evaluation

- Dataset001_BHSD multiclass, fold 0, 256 x 256 patches, three slices.
- Standard nnU-Net optimizer/schedule and Dice + CE loss.
- Maximum 1000 epochs; minimum 300; EMA foreground-Dice patience 100.
- Select `checkpoint_best.pth`; validate with `--val_best --npz`.
- Rank only by full-case `validation/summary.json` foreground Dice.
- Online EMA is a checkpoint-selection signal and must not be compared with
  summary Dice.

F1/F2 are promising only if they exceed controlled C3 by a material margin
(approximately 0.01 foreground Dice) without a major subtype regression.
Fold-0 outcomes remain screening evidence; the winning fusion must then be
confirmed across multiple seeds and folds.

## Compute-cost rule

Every array arm runs `scripts/profile_25d_compute.py` before training and saves
the same `compute_profile.json` in experiment metadata and the fold result
tree. It records total/adapter parameters, batch-1 forward latency, allocated
GPU-memory peak, and backbone passes per prediction. Full-run wall time and
sampled training memory remain in `stage_metrics.csv` and `run_timing.json`.

After all arms finish, use `scripts/summarize_controlled_screens.py --screen
fusion` to combine full-case Dice and cost. A complex fusion is not advanced
merely because it is larger: F1/F2 must provide at least 0.01 foreground-Dice
gain over C3, avoid a class decrease greater than 0.02 relative to C3, and
remain on the Dice/runtime/memory/parameter Pareto frontier. Resource ratios
are also reported against C0.

## Gadi submission

```bash
cd /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet
git pull --ff-only origin main
cd /scratch/ke17/bhsd-nnunet/logs
qsub /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet/hpc/gadi/train_25d_fusion_screen_fold0.pbs
```

The array indices are C0, C1, C2, C3, F1, and F2 respectively.

## Controlled A8 multi-seed comparator

The original A8 score (`0.295321`) came from the earlier screening seed policy
and is not used as a controlled seed-3407 observation. C3 is the deterministic
replication of the same `axial_slice_conv` implementation and supplies the
completed controlled seed-3407 reference (`0.244792`). To compare this axial
model fairly with E2 across initialization seeds without rerunning A8 or C3,
two isolated C3-derived trainers add model seeds 1234 and 5678. Their data seed
remains 1003410 and all other training and validation fields match C3.

Submit only the two additional controlled A8 comparator jobs:

```bash
qsub hpc/gadi/train_25d_axial_multiseed_fold0.pbs
```

After the controlled A8 jobs and the E0/E2 jobs complete, compare the three
models across seeds 3407, 1234, and 5678:

```bash
python3 scripts/summarize_axial_multiseed_comparison.py \
  --results-root "$nnUNet_results" \
  --output-dir "$BHSD_RESULTS_DIR/controlled_a8_e0_e2_multiseed_summary"
```

This is a direct model-performance comparison, not a matched mechanism-effect
estimate: E0 is the direct control for E2, while C1 is the matched capacity
control for C3. The primary A8-versus-E2 comparison remains the best-checkpoint
nnU-Net `validation/summary.json` foreground Dice. Secondary case-macro views
remain separately labelled, and online EMA is not aggregated.

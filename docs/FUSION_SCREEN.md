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
augmentation. This controls the experiment's stochastic inputs.

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

## Gadi submission

```bash
cd /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet
git pull --ff-only origin main
cd /scratch/ke17/bhsd-nnunet/logs
qsub /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet/hpc/gadi/train_25d_fusion_screen_fold0.pbs
```

The array indices are C0, C1, C2, C3, F1, and F2 respectively.

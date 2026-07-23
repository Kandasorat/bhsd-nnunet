# Single-pass symmetric reliability screen (E0-E2)

## Question

D6 produced the strongest D-screen fold-0 result, but it averages original and
neighbor-swapped predictions and therefore uses two complete backbone passes.
This screen asks whether the useful invariance can be retained with one pass and
a small input adapter.

## Mathematical constraint

An ordered raw input `[previous, center, following]` is not invariant to swapping
the two neighbors. An invariant gate alone cannot make the complete one-pass
network invariant while this ordered path remains. E0-E2 therefore feed the
backbone the symmetric basis

`[center, (previous + following) / 2, |previous - following| / 2]`.

The descriptor computes stable context and disagreement:

`S = (F_previous + F_center + F_following) / sqrt(3)`

`U = (|F_previous - F_center| + |F_following - F_center|) / 2`.

E2 uses `g = sigmoid(h([F_center, S, U]))` and a zero-initialized projected
residual based on `g * (S - F_center)`. These quantities are exactly unchanged
when previous and following slices are exchanged.

## Controlled arms

| Arm | Residual source | Purpose |
| --- | --- | --- |
| E0 | `g([F_center, F_center, 0]) * F_center` | Symmetric representation and adapter-capacity control |
| E1 | `g([F_center, S, 0]) * (S - F_center)` | Tests low-frequency context without a disagreement input |
| E2 | `g([F_center, S, U]) * (S - F_center)` | Tests disagreement-aware reliability gating |

All arms use the same nnU-Net 2D backbone, three input slices, fold 0 split,
loss, optimizer, seed pair, early stopping, descriptor width, and one backbone
pass. The residual projection is zero initialized, so all arms begin from their
shared symmetric input representation.

## Required interpretation

- E1 versus E0: value of symmetric low-frequency context.
- E2 versus E1: value of adding explicit disagreement to a capacity-matched gate.
- E2 versus D6: accuracy-cost trade-off, not a pure module ablation, because the
  input representation and number of backbone passes differ.
- Do not call the slice-index contrasts physical derivatives. Analyse effects by
  lesion volume and hemorrhage class with `scripts/analyze_case_level_effects.py`.

## Gadi commands

Train E0-E2:

```bash
qsub hpc/gadi/train_25d_symmetric_reliability_fold0.pbs
```

Run the no-retraining direction probe for D1, D5, and D6:

```bash
qsub hpc/gadi/evaluate_spectral_direction_probe_fold0.pbs
```

After E0-E2 finish, summarize Dice together with recorded parameter, runtime,
and memory cost:

```bash
python3 scripts/summarize_controlled_screens.py \
  --screen symmetric \
  --results-root "$nnUNet_results" \
  --metadata-root "$BHSD_RESULTS_DIR" \
  --output "$BHSD_RESULTS_DIR/symmetric_e0_e2_summary.csv"
```

Generate the no-retraining case-level lesion-volume and class tables
from the three best-checkpoint validation summaries:

```bash
python3 scripts/analyze_case_level_effects.py \
  --metrics E0="$nnUNet_results/Dataset001_BHSD/nnUNetTrainer_25D_SymmetricE0Control__nnUNetPlans__2d/fold_0/validation/summary.json" \
  --metrics E1="$nnUNet_results/Dataset001_BHSD/nnUNetTrainer_25D_SymmetricE1LowPass__nnUNetPlans__2d/fold_0/validation/summary.json" \
  --metrics E2="$nnUNet_results/Dataset001_BHSD/nnUNetTrainer_25D_SymmetricE2ReliabilityGate__nnUNetPlans__2d/fold_0/validation/summary.json" \
  --reference-model E0 \
  --ground-truth-dir "$nnUNet_preprocessed/Dataset001_BHSD/gt_segmentations" \
  --output-dir "$BHSD_RESULTS_DIR/symmetric_e0_e2_case_analysis"
```

The analysis keeps three views separate: model-specific finite-class case
macro Dice, paired Dice restricted to classes present in ground truth, and
false-positive case fractions for ground-truth-absent classes. It also writes
`SHA256SUMS.txt` for transfer verification. These case-level views must not be
substituted for the nnU-Net `foreground_mean.Dice` primary screen result.

## Pre-specified multi-seed confirmation

Only E0 and E2 advance. Two additional model seeds, 1234 and 5678, are paired
within fold 0. The data seed remains 1003410 for all six runs (including the
completed seed 3407 pair), so the confirmation varies model initialization
without changing the epoch-indexed augmentation policy. Every new config uses
an isolated trainer class and nnU-Net output namespace; it cannot resume from
or overwrite the completed seed 3407 result. E1 is not included.

Submit the four new runs:

```bash
qsub hpc/gadi/train_25d_symmetric_multiseed_fold0.pbs
```

After all four jobs complete, estimate the paired E2-minus-E0 effect across the
three model seeds:

```bash
python3 scripts/summarize_symmetric_multiseed.py \
  --results-root "$nnUNet_results" \
  --output-dir "$BHSD_RESULTS_DIR/symmetric_e0_e2_multiseed_summary"
```

The primary endpoint is the best-checkpoint nnU-Net
`validation/summary.json` `foreground_mean.Dice`. The script labels both the
model-specific finite-class case macro and the common-support,
ground-truth-present class macro as separate secondary endpoints; it does not
read or aggregate online EMA Dice. This distinction matters because nnU-Net
records Dice as NaN when both truth and prediction are empty, so model-specific
finite supports can differ. With only three model seeds on one fold, report the
individual paired deltas, their mean, sample SD, range, and sign consistency;
do not treat this as a substitute for folds 1-4.

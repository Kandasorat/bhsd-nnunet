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
  actual z-spacing and lesion volume with `scripts/analyze_case_level_effects.py`.

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

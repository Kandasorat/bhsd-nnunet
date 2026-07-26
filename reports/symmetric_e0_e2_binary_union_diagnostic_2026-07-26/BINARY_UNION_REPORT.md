# Post-hoc binary-union diagnostic of E0/E2 multiclass predictions

**This is not a binary-trained model result.** It measures what the existing multiclass models become after all five hemorrhage labels are treated as one foreground class.

## Primary hard-union results

| Seed | Model | Mean Dice | SD | Mean HD95 (mm) | FP (mL) | FN (mL) | Recall | Complete misses |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3407 | E0 | 0.6437 | 0.2591 | 30.17 | 5.565 | 7.800 | 0.6633 | 0 |
| 3407 | E2 | 0.6244 | 0.2640 | 34.02 | 4.701 | 9.227 | 0.6351 | 0 |
| 1234 | E0 | 0.6413 | 0.2706 | 28.40 | 4.026 | 8.679 | 0.6243 | 0 |
| 1234 | E2 | 0.6276 | 0.2554 | 37.88 | 5.782 | 8.438 | 0.6645 | 0 |
| 5678 | E0 | 0.6410 | 0.2439 | 37.05 | 5.886 | 8.140 | 0.6816 | 0 |
| 5678 | E2 | 0.6437 | 0.2439 | 34.42 | 5.763 | 8.028 | 0.6795 | 0 |

## Fixed-threshold soft-union results

`P(ICH)=1-P(background)` with a pre-specified threshold of 0.5; no validation threshold search was performed.

| Seed | Model | Mean Dice | Mean HD95 (mm) | FP (mL) | Recall |
|---:|---|---:|---:|---:|---:|
| 3407 | E0 | 0.6443 | 30.18 | 5.625 | 0.6657 |
| 3407 | E2 | 0.6261 | 34.02 | 4.739 | 0.6381 |
| 1234 | E0 | 0.6430 | 28.38 | 4.074 | 0.6278 |
| 1234 | E2 | 0.6293 | 37.99 | 5.851 | 0.6683 |
| 5678 | E0 | 0.6419 | 37.05 | 5.938 | 0.6842 |
| 5678 | E2 | 0.6447 | 34.32 | 5.880 | 0.6832 |

## Small-lesion diagnostic

The pre-existing whole-case thresholds are retained. Fold 0 has only four `<1 mL` patients, so these values are descriptive.

| Seed | Model | n | Mean hard-union Dice | Complete misses |
|---:|---|---:|---:|---:|
| 3407 | E0 | 4 | 0.2454 | 0 |
| 3407 | E2 | 4 | 0.1374 | 0 |
| 1234 | E0 | 4 | 0.2810 | 0 |
| 1234 | E2 | 4 | 0.1849 | 0 |
| 5678 | E0 | 4 | 0.1757 | 0 |
| 5678 | E2 | 4 | 0.1906 | 0 |

## Paired E2-versus-E0 result
The three-seed average paired hard-union Dice delta is -0.01010; patient-bootstrap 95% CI [-0.02585, +0.00569]. This evaluates E2 versus E0, not 2.5D versus 2D.

## Subtype-confusion recovery
Among hard-union true-positive foreground voxels, the mean fraction assigned to the wrong hemorrhage subtype is 12.0% for E0 and 13.3% for E2. These voxels become correct only after subtype labels are collapsed.

## Interpretation
If hard-union Dice is strong and reproducible while multiclass subtype performance remains weak, the principal bottleneck is subtype separation rather than foreground localization. If hard-union remains weak—especially for `<1 mL` cases—the binary route does not automatically solve the localization problem.

The result cannot be reported as binary-model performance, cannot show what binary loss would learn, and cannot establish that 2.5D helps because both E0 and E2 are three-slice models.

## Decision for the next experiment
A binary training experiment is justified only as a new controlled question: binary-trained center-only 2D versus binary-trained simple symmetric 3-slice 2.5D, followed by multi-fold confirmation if the fold-0 seed effect is stable.

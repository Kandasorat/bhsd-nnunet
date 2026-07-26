# E0/E2 multi-seed complementarity and GT-oracle analysis

**Locked decision: `E0_E2_COMPLEMENTARITY_STOP`**

## 1. Scope
Only existing fold-0 hard predictions, exported probabilities, and the 39 validation GT cases were analyzed. No training, inference, resampling, or data expansion was performed.

## 2. Asset and provenance audit
All six result sets contain 39 hard predictions, 39 NPZ files, 39 properties files, both checkpoints, summary, debug, timing, and profiling records. All validations record `--val_best`, so `checkpoint_best.pth` is the validated checkpoint.

## 3. Case-ID join
All joins were explicit by case ID. The six prediction sets and GT contain the same 39 IDs with no duplicates, missing cases, or extras.

## 4. Geometry, labels, and voxel volume
Every prediction matches GT shape and affine. Labels are restricted to 0-5. Affine-determinant and header-zoom voxel volumes agree within tolerance.

## 5. Seed-3407 reproduction
Reproduction status: `passed`. E0 absent FP=70/113; E2 absent FP=58/113; maximum GT-present Dice difference from the historical analysis=0.

## 6. Metric separation
nnU-Net `foreground_mean.Dice`, class-balanced GT-present macro, pooled patient-class Dice, and online EMA are not interchangeable. The decision uses class-balanced GT-present metrics.

## 7. GT-present performance
| Seed | E0 class-balanced present | E2 class-balanced present | E2-E0 |
|---:|---:|---:|---:|
| 3407 | 0.482249 | 0.452128 | -0.030122 |
| 1234 | 0.451579 | 0.502839 | +0.051260 |
| 5678 | 0.490548 | 0.507325 | +0.016777 |

## 8. Absent-class false positives
| Seed | E0 any-FP pairs | E2 any-FP pairs | Denominator |
|---:|---:|---:|---:|
| 3407 | 70 | 58 | 113 |
| 1234 | 55 | 82 | 113 |
| 5678 | 75 | 92 | 113 |

## 9. Cross-seed direction stability
Median pairwise Spearman rho=-0.2020; strong-effect three-seed unanimous practical direction=11.8% (n=76).

## 10. Practical sign definition
A delta >=+0.01 favors E2, <=-0.01 favors E0, and intermediate values are neutral. Majority sign is not treated as stability.

## 11. Patient-cluster bootstrap
Primary three-seed class-balanced delta=+0.012639; percentile 95% CI [-0.008684, +0.037421], valid=9987, invalid=13. Models seeds are retained jointly within sampled patients.

## 12. Patient-level sign flip
Two-sided p=0.403360; one sign was applied jointly to all classes and seeds for each patient.

## 13. Case-class GT oracle
| Seed | Oracle headroom vs best single |
|---:|---:|
| 3407 | +0.038182 |
| 1234 | +0.041601 |
| 5678 | +0.033241 |
Mean headroom=+0.037674; sample SD=0.004203; patient-bootstrap 95% CI [+0.027266, +0.047027].

## 14. Leave-one-seed-out selector
Positive held-seed gains=0/3; mean gain vs the stronger complete single model=-0.031738. This selector uses the same patients' GT from the other two model seeds and is not deployable or an unseen-patient test.

## 15. Presence oracle
The one-vs-rest presence oracle removes predictions only when that class is GT-absent. It preserves every GT-present binary Dice exactly and is a diagnostic of summary sensitivity to absent-class FP, not model performance.

## 16. Slice selectors
The exact label `exploratory_highly_optimistic_slice_gt_greedy_selector` is used. Soft selectors were included after exact NPZ restoration verification. These GT-guided slice results are not a strict volume oracle.

## 17. Oracle invariants
For every seed and class, the case-class oracle is at least as high as both E0 and E2 on corresponding GT-present support; ties use E0 deterministically.

## 18. Decision rule evaluation
GO checks passed: 5/9. STOP triggers met: 4/8. Final decision is `E0_E2_COMPLEMENTARITY_STOP`.

## 19. Interpretation boundary
E0 and E2 both use a full symmetric three-slice backbone. This comparison cannot establish whether neighboring context helps relative to center-only 2D, whether the center residual helps, or whether a learned selector generalizes.

## 20. Validation limitation
There are only 39 fold-0 validation patients, and these cases participated in checkpoint selection. There is no independent test set.

## 21. Uncertainty limitation
The patient bootstrap preserves within-patient classes and seeds but does not include fold uncertainty. Three model seeds are not independent cohorts.

## 22. Oracle limitation
All GT oracles and GT-guided selectors are undeployable conditional diagnostics. They must not be reported as achieved model scores or SOTA comparisons.

## 23. Research claim boundary
The result concerns only repeatable E0/E2 complementarity on fold 0. It does not stop all 2.5D research and does not support a SOTA claim.

## 24. Reproducibility
Configuration, manifests, exact case lists, generated tables, tests, and SHA-256 checksums are included in this directory. Historical seed-3407 outputs were read-only and not regenerated.

## 18-point executive summary

- The locked outcome is `E0_E2_COMPLEMENTARITY_STOP`.
- Six required E0/E2 result sets are complete.
- All sets contain the same 39 fold-0 cases.
- Hard-prediction geometry matches GT exactly within the audit tolerance.
- All observed labels are valid integers 0-5.
- Validation provenance points to `checkpoint_best.pth`.
- Locked nnU-Net summary scores reproduce within 1e-6.
- Seed 3407 absent-FP counts reproduce exactly at 70/113 and 58/113.
- All 390 historical E0/E2 seed-3407 class-case count rows reproduce.
- GT-present Dice is the primary support for complementarity decisions.
- Mean strict case-class oracle headroom is +0.037674.
- LOSO selector mean gain is -0.031738.
- Median cross-seed patient-class delta Spearman rho is -0.2020.
- Strong-effect unanimous practical direction is 11.8%.
- Absent-class FP diagnostics are kept separate from GT-present delineation.
- Presence-oracle numbers are diagnostics, not model scores.
- Slice GT-greedy numbers are highly optimistic and not strict volume-oracle results.
- The analysis cannot attribute effects to neighbor context versus center residual because both arms are three-slice models.

## Exactly one next-step recommendation
Do not design or train another E0/E2 selector from these fold-0 oracle diagnostics; use the locked decision to close this branch and, in a separate planning step, choose a publication question that can be tested against an explicit center-only 2D baseline with multi-fold or independent-test confirmation.

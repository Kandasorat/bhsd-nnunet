# BHSD comparison-validity and Stage3 readiness audit

Audit date: 2026-07-26  
Repository inspected: `C:\Users\92127\OneDrive - UNSW\project_linpeng\code` at `0c660e0786386a7dbe07c8366068e2c9455aeb7e`  
Authority: canonical `PROJECT_HANDOFF.md`, except its obsolete “Immediate next work” section  
Locked current E decision: `E0_E2_COMPLEMENTARITY_STOP`

## Executive conclusion

The current standard 2.5D data path is implemented correctly for consecutive slice selection, center-slice supervision, clamped boundary replication, synchronized geometry, patient-level split isolation, validation case IDs, and full-case summary reading. Synthetic tests, 20 selected real image/GT cases, 20 prediction/GT pairs, and 52 historical validation summaries found no failure in those areas.

That implementation correctness does **not** make every historical comparison causal or publication-grade. Formal five-fold 2D/3D results are pipeline benchmarks, not isolated dimensionality experiments. A0 and A1-A8 are fold0 exploratory screens with unverified child/worker seeding. C/F improved seed propagation but predate the separate data-seed reset. D and E are the strongest controlled fold0 ablations; E0/E2 additionally have three model seeds, but both arms consume three slices and therefore cannot determine whether real neighbours add value over a center-only model.

The proposed B/R0/R1 design is the correct causal structure for separating baseline ability, adapter/capacity benefit, and true-neighbour benefit. It is not executable yet. The Stage3 wrapper/loader/evaluator and its required tests do not exist, and the historical intensity pipeline would destroy exact R0 duplication if duplication occurred before standard channel-wise intensity augmentation. The generic evaluation helper also does not implement physical HD95 or patient-cluster statistics. These are pre-training blockers, not reasons to alter historical results.

## 1. Files and directories inspected

The audit read or indexed:

- canonical `C:\Users\92127\OneDrive - UNSW\project_linpeng\PROJECT_HANDOFF.md` in full;
- Git status/history and historical revisions `1167efe`, `fcdda534`, `4fa9572`, `6c1653d`, `e60d41f`, `0c660e0`;
- all relevant `configs/*.yaml` for 2D, 3D, A0, A1-A8, C/F, D, E, and controlled multiseed arms;
- `nnunet25d/common/dataloader_25d.py`;
- `nnunet25d/baseline/trainer_25d.py` and historical versions of its seed logic;
- `scripts/run_experiment.py`, `nnunet25d/common/early_stopping.py`, and the installed nnU-Net evaluator/logger/transform code;
- `nnunet25d/attention/{unified_slice_adapters,spectral_slice_fusion,symmetric_reliability_fusion}.py` through code review and existing verifiers;
- `evaluation/metrics.py`, `evaluation/run_evaluation.py`, `evaluation/statistical_tests.py`, and `scripts/evaluate_binary_segmentation.py`;
- `docs/ATTENTION_REPRODUCTION_POLICY.md`, `ATTENTION_MODULE_SCREEN.md`, `FUSION_SCREEN.md`, `SPECTRAL_SLICE_SCREEN.md`, and `SYMMETRIC_RELIABILITY_SCREEN.md` in full;
- locked `Dataset001_BHSD/splits_final.json`, raw/preprocessed case properties, 20 raw image/GT cases, and 20 E0 prediction/GT pairs;
- 52 `validation/summary.json` trees across formal 2D/3D, A0, A1-A8, C/F, D, E, and added multiseed results;
- the latest `ORACLE_ANALYSIS_REPORT.md`, `oracle_decision.json`, manifest, tests, and metric code;
- the posthoc binary-union diagnostic only to audit metric definitions; it was not recomputed;
- requested local backups under `D:\BHSD_server_backups\SRG_SF_direction_results_2026-07-24` and `D:\BHSD_server_backups\multiclass_25d_controlled_screens_2026-07-22`;
- controlled multiseed trees under `C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups\controlled_multiseed_fold0_2026-07-24`.

Gadi was not needed: local result trees were complete for this audit. No server write or job query was performed.

## 2. Reproduction audit results

### Three-slice construction and supervision

`nnUNetDataLoader25D._get_slice_indices` clamps `center+offset` to `[0,Z-1]`; `_stack_input_slices` concatenates those slices in ascending offset order. `generate_train_batch` uses the clamped crop's first dimension as the center and crops only `seg[:,center_slice]` as the target. Full-case validation constructs the same triplets for every center slice.

The unique-valued synthetic volume produced:

- z=0: input `[0,0,1]`, target 0;
- z=2: input `[1,2,3]`, target 2;
- z=last: input `[last-1,last,last]`, target last.

There is no evidence that historical A/C/D/E runs trained against a neighbour GT or reversed slice order.

### Augmentation

The loader calls one transform object with a multichannel image tensor and its center segmentation. The nnU-Net `SpatialTransform` and `MirrorTransform` therefore sample one geometry for all three slices and GT. Across 24 randomized synthetic spatial trials, all image channels remained geometrically identical and their thresholded alignment Dice with GT was 1.0.

Intensity behavior is different. Of the actual transforms, Gaussian noise has `synchronize_channels=True`; Gaussian blur, multiplicative brightness, contrast, simulated low resolution, and both gamma transforms have `synchronize_channels=False`. Identical channels became nonidentical in 26 of 40 fixed-seed full-pipeline trials. This is the standard historical behavior and applies comparably to all existing three-channel arms. It is nevertheless incompatible with Stage3 R0 if `[x0,x0,x0]` is constructed before augmentation, because the residual branch would no longer receive exact duplicates.

### Split and case identity

The split SHA-256 is `A7F3088C3195273FEFFAA06A99E9A8F2C62F6AEB0AC5DC97A8498D1D5C55BEEA`. Each fold has disjoint train/validation patient IDs; the five validation lists cover all 192 cases exactly once. All 52 audited summaries contain exactly the expected fold validation IDs with no duplicates. Binary summaries match the same case split; the binary preparation PBS explicitly copied Dataset001 splits into Dataset002.

The 20 real-case sample included the ten smallest-z cases (minimum 24 slices) and additional cases with up to five present hemorrhage classes. All image/label geometries matched and all labels were in 0-5. The 20 E0 prediction/GT pairs matched case ID, size, spacing, origin, direction, and label range. The locked oracle independently checked all 39 fold0 cases for all six E0/E2 seed result sets.

### Checkpoint and metric provenance

All 52 audited result trees contain both `checkpoint_best.pth` and `checkpoint_final.pth`. The locked configs request `--val_best`, and every summary's `foreground_mean.Dice` exactly equals the arithmetic mean of its stored foreground class means. Historical headline tables and the oracle read those summaries rather than online EMA.

Online EMA remains a selection signal: it is computed from pooled hard counts over sampled validation patches, averaged across classes, then smoothed with `0.9/0.1`. It is not full-volume patient Dice. The validation patients are also used to select the best checkpoint; reported cross-validation values therefore include checkpoint-selection optimism and are not independent-test estimates.

## 3. Seed audit

- Formal 2D/3D, A0, and A1-A8 configs name seed 3407, but historical debug files do not establish a child model seed or data seed; legacy four-worker augmentation used `seeds=None`. Their results remain valid pipeline/screen records, not same-seed paired effects.
- C/F pass seed 3407 into the training child and force single-thread deterministic augmentation, but commit `fcdda534` applies one RNG stream and predates the separate data-seed reset. Different parameter counts can consume different initialization draws and shift later data/augmentation draws. C/F are controlled screens, not exact paired capacity experiments.
- D and E pass a model seed plus fixed data seed 1003410 to the child, reset data streams after initialization, use epoch-indexed seeds, and run single-thread augmentation. CUDA determinism is best effort with `warn_only=True`, so “controlled” does not mean bitwise guaranteed on every device.
- E0/E2 and controlled axial added seeds 1234/5678 under the D/E policy. Those seeds quantify initialization sensitivity on the same 39 patients; they are not independent cohorts.

## 4. Metric and statistics findings

The required metric families are defined separately in `metric_definition_audit.csv`. Two code-level cautions matter for future work:

1. `evaluation/metrics.py` returns Dice 0 for both-empty masks and computes maximum Hausdorff on voxel-index coordinates. That value is neither physical HD95 nor spacing-aware. It was not used for the formal nnU-Net headline tables or locked oracle, but it must not be reused for Stage3 HD95/NSD.
2. `evaluation/statistical_tests.py` pairs whatever rows are supplied and can treat patient-class rows as independent observations. It has no patient-cluster bootstrap or sign flip. The locked oracle correctly clusters by patient; Stage3 must use the same principle.

No locked historical table was found directly comparing online EMA, nnU-Net summary Dice, and present-macro Dice as if they were the same endpoint. Some prose and direction-probe tables use different aggregations; their labels and the handoff explicitly prohibit numerical interchange. Binary and multiclass results are also separate learned tasks: a union of multiclass predictions is a posthoc diagnostic, not the result of a binary-trained objective.

## 5. Validity of historical comparisons

The complete row-level audit is in `comparison_claim_matrix.csv`.

| Evidence group | Validity | What it can answer |
|---|---|---|
| Multiclass/binary 2D vs 3D five-fold | `FORMAL_COMPARISON` | Which complete nnU-Net pipeline has higher locked cross-validation validation performance. It cannot attribute the difference only to dimensionality. |
| Multiclass/binary A0 vs 2D fold0 | `EXPLORATORY_SCREEN` | Whether naive stacking automatically helped that historical fold. It cannot isolate neighbour information because input channels, parameters, stochastic trajectory, and stopping differ. |
| A1-A8 | `EXPLORATORY_SCREEN` | Candidate ranking under a harmonized nnU-Net fold0 screen. Capacities differ and the runs are not source-faithful paper reproductions. |
| C0-C3/F1-F2 | `CONTROLLED_SCREEN` | Fold0 plausibility under improved child seeding/single-thread policy. Capacities and the historical post-init RNG stream are not fully matched. |
| D0-D6 | `CONTROLLED_SCREEN` | Within-series fold0 spectral ablations, with exact capacity matching only for declared subsets; D6 has two backbone passes. |
| E0/E1/E2 seed3407 | `CONTROLLED_SCREEN` | Exact capacity-matched differences among symmetric three-slice mechanisms on one seed/fold. |
| E0/E2 three seeds | `CONTROLLED_SCREEN` | Initialization stability of E2 versus E0 on fold0. It cannot answer neighbour versus center-only. |
| E0/E2 GT oracle | `DIAGNOSTIC_ONLY` | Undeployable complementarity headroom and failure of tested stability/LOSO criteria. It is not a model score. |
| Direct numerical comparison to published CSAM/CSA-Net/ACSFormer/SDC-UNet | `INVALID_COMPARISON` | Only qualitative prior-art positioning is allowed. |

The locked evidence says E2 is not stable: its nnU-Net summary delta versus E0 is positive for seed3407 but negative for seeds1234 and5678; present-macro direction is also heterogeneous and the patient-cluster CI crosses zero. Controlled axial is likewise unstable. The `E0_E2_COMPLEMENTARITY_STOP` decision is supported and must not be used to generate another E selector.

## 6. Claims that may be written in a paper

- Under the locked five-fold nnU-Net pipelines, 3D produced higher multiclass validation Dice than 2D, including higher mean Dice for every subtype.
- Binary union segmentation is an easier and different endpoint; its 2D/3D ordering does not determine multiclass subtype performance.
- Naive three-slice A0 did not automatically improve multiclass fold0, but folds1-4 are still missing for a formal A0 benchmark.
- Harmonized fold0 fusion/attention screens show strong dependence on module, seed policy, and control definition; added complexity did not consistently beat simple controls.
- D/E results support reporting heterogeneity, negative transfer, absent-class FP behavior, and compute/accuracy trade-offs on fold0.
- E0/E2 complementarity was not stable across three seeds on fold0, and a GT-guided oracle did not translate into a successful deployable selector.
- The current work motivates a preregistered frozen-center R1-versus-R0 experiment; it does not yet demonstrate neighbour value.

Each claim must say “cross-validation validation” or “fold0 screening” as applicable and must not imply an independent test set.

## 7. Claims that are forbidden

- “3D wins because dimensionality/context alone is causal.”
- “2.5D is ineffective on BHSD” or “all 2.5D routes are disproven.”
- “E2/axial is a stable winner” or “D6 is the best efficient one-pass model.”
- “E0/E2 proves true neighbours help” — both use three-slice inputs.
- Reporting a GT oracle, presence oracle, or GT-guided slice selector as achieved model performance.
- Treating three random seeds as independent patients or treating patient-class/slice rows as independent cohorts.
- Calling A2/A5 or any harmonized adapter an official/source-faithful CSAM or CSA-Net reproduction.
- Directly comparing binary-trained Dice with multiclass foreground Dice, or using a posthoc binary union as a substitute for binary training.
- Calling the generic voxel-index maximum Hausdorff “HD95 mm.”
- Claiming SOTA from fold0 or from checkpoint-selected cross-validation without an independent test.

## 8. Relationship to published 2.5D work

There is no direct replication conflict.

- [CSAM](https://arxiv.org/abs/2311.04942) applies semantic, positional, and slice attention on deep features and its released protocol uses substantially different slice count, network, optimizer, loss, input size, and tasks. A2 is a small nnU-Net input-adapter gate with source components omitted.
- [CSA-Net](https://arxiv.org/abs/2405.00130) uses previous/center/next inputs but an R50-ViT-B/16-based design, pretrained weights, and published MRI tasks/protocol. A5 is a reduced descriptor-grid nnU-Net adaptation.
- [ACSFormer](https://www.sciencedirect.com/science/article/pii/S0031320325015584) evaluates a different adaptive CNN-transformer framework on INSTANCE and other anisotropic datasets with a separate train/validation/test protocol. Current BHSD fold0 screens neither reproduce nor refute it.
- [SDC-UNet](https://doi.org/10.15938/j.jhust.2025.03.001) establishes slice-difference processing as prior art for ICH CT. D-series uses a different orthogonal slice-index representation and is an ablation, not a replication.

Architecture fidelity, training-protocol fidelity, dataset, label task, split, checkpoint selection, and metric must all match before a published result can be called reproduced or contradicted. None currently do.

## 9. Does B/R0/R1 answer neighbour value?

Conceptually yes:

- B fixes the original center-only capability;
- R0-B measures adapter/capacity/optimization benefit without new anatomy;
- R1-R0 isolates true-neighbour information if the only changed tensor is the neighbour content;
- R1-B is the deployable total effect.

The comparison is valid only if the Stage3 protocol is implemented exactly: same frozen B hash, one backbone pass, identical residual parameters, paired seeds/data, exact R0 duplicates after augmentation, synchronized three-slice intensity parameters, center-only GT, and patient-cluster statistics. The preregistration fixes these requirements in `stage3_preregistered_protocol.md`.

## 10. Stage3 blockers

1. No B/R0/R1 wrapper or Stage3 loader exists; the task explicitly forbids implementing it now.
2. None of the 25 required Stage3 tests has been run because there is no implementation.
3. The historical intensity transform is mostly channel-independent; Stage3 needs a locked shared-intensity path and must prove R0 exact duplication at the residual boundary.
4. B checkpoint hashes for folds1-4 have not been frozen into a Stage3 manifest.
5. A physical-space HD95/NSD and lesion-component evaluator with the preregistered empty rules is not implemented; the generic helper is unsuitable.
6. A patient-cluster confirmatory analysis script is not frozen; the generic paired-test helper is unsuitable.
7. Exact parameter/FLOP/memory/latency/pass-count outputs for the future wrapper do not exist.

The incomplete A0 folds1-4 remain a publication benchmark gap but are not evidence that the B/R0/R1 causal design is wrong. If A0 is completed, its legacy protocol must be isolated at code-equivalent revision `3c2c38b`; current HEAD has materially different seed behavior and cannot be silently called the same historical protocol.

## Final status

`STAGE3_BLOCKED`

## Exactly one next-step recommendation

After explicit approval, perform one code-only Stage3 implementation pass that adds only the locked B/R0/R1 wrapper, shared-intensity loader path, physical metric evaluator, patient-cluster analysis skeleton, and all tests in `stage3_required_tests.md`; freeze hashes and stop again before any training or PBS submission.

# BHSD fixed1000 confirmatory protocol

Status: preregistered; formal training is prohibited until the Gadi V100 readiness decision is `PASS`.

## Scope

This protocol uses only the 192 pixel-annotated BHSD cases and the existing five folds in `splits_final.json` (SHA-256 `A7F3088C3195273FEFFAA06A99E9A8F2C62F6AEB0AC5DC97A8498D1D5C55BEEA`). It excludes unlabeled data, weak labels and external data.

Exactly two arrays are authorized:

- `fixed1000_core_multiclass_15`: 2D, A0 and 3D, folds 0–4, model seed 3407.
- `fixed1000_fold0_diagnostic_12`: C1, C2, D0 and D1, fold 0, model seeds 3407, 1234 and 5678.

No E0/E2, controlled axial, D2–D6, other A/F variants, Stage3, binary experiment or new architecture is authorized. Active CSA-Net/CSAM jobs and directories are outside this protocol and must not be modified.

## Locked training schedule

All 27 runs use 1000 epochs, 250 training iterations per epoch, 50 validation iterations per epoch, SGD with learning rate 0.01, momentum 0.99, Nesterov enabled and weight decay `3e-5`. PolyLR has horizon 1000 and exponent 0.9. Performance-based early stopping is absent from the class MRO and cannot be enabled by an environment variable.

All runs use model-specific locked model seeds and a locked data seed 1003410. `nnUNet_n_proc_DA=0`, deterministic CUDA settings and epoch-scoped data RNG resets make the initialization and augmentation stream explicit. Core 2D/A0 uses batch 12 and 3D uses the existing batch 2.

The primary checkpoint is `checkpoint_final.pth`. `checkpoint_best.pth` is sensitivity-only. Full-case validation and probabilities are written separately to `validation_final/` and `validation_best_sensitivity/`. Existing result folders cause a fail-closed error; resume and overwrite are disabled.

## Locked architectures

- A0: `[previous, centre, next]`, centre-slice supervision, replicated z boundaries, fixed consecutive slices.
- C1: historical matched adapter-capacity control.
- C2: historical controlled CSA centre-neighbour mechanism.
- D0: historical centre-descriptor capacity control.
- D1: historical low-frequency persistence mechanism.

The fixed trainers reuse the historical 2.5D data, inference and adapter implementations as unbound methods/components but do not inherit the historical early-stopping base. Local state-dict and initialization audits must remain exact.

## Readiness and submission

The 18 static checks, local architecture audit, Gadi V100 single-batch forward/backward validation, checkpoint round-trip, resource projection, scratch quota, namespace absence and immutable Git checks must all pass. Fold-0 Dice is not a readiness criterion.

The two formal arrays may be submitted only from the clean Git commit named by `FIXED1000_EXPECTED_COMMIT`. No partial performance inspection may change pending tasks. Infrastructure-only monitoring is allowed. A failed configuration or code gate stops the affected array; silent reruns are prohibited.

## Preregistered analysis

Core comparisons are A0−2D, 3D−2D and A0−3D using five-class foreground macro Dice, patient-level paired OOF differences, GT-present macro and per-class Dice, absent-class FP cases/volume, lesion F1, small-lesion recall, HD95/NSD and compute measures.

Fixed-vs-adaptive training-schedule sensitivity is declared if the A0−2D sign changes, its magnitude differs by at least 0.01, or at least two folds reverse the A0/2D direction. Historical baseline seed provenance that is not embedded in a checkpoint is a limitation of fixed-vs-adaptive interpretation, not of the internally seed-matched fixed matrix.

Fold-0 diagnostic escalation is evaluated separately for C2−C1 and D1−D0. It requires mean GT-present gain at least +0.01, at least two of three seeds positive, no absent-FP-driven apparent gain, no preregistered severe per-class regression, and mechanism-consistent compute. Passing permits later folds but is not itself a formal cross-fold performance result.


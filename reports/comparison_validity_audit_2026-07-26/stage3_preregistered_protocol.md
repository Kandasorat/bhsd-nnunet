# Stage 3 preregistered protocol: frozen-center residual test

Date frozen: 2026-07-26  
Scientific task: `Dataset001_BHSD` five-class ICH segmentation  
Current authorization: design and tests only; no implementation, training, inference, PBS submission, commit, or push is authorized by this document.

## 1. Primary question and estimands

The sole question is whether true adjacent slices provide a reproducible benefit after separating the original center-only model and the benefit of adding a trainable adapter.

For augmented center slice `x0`, preceding slice `x-1`, following slice `x+1`, and a frozen 2D model `f_theta`:

```text
z_B  = f_theta(x0)
z_R0 = z_B + delta_phi(x0, x0, x0, stopgrad(z_B))
z_R1 = z_B + delta_phi(x-1, x0, x+1, stopgrad(z_B))
```

The three predeclared effects are:

- `R0-B`: extra capacity, adapter training, and adapter-optimization effect;
- `R1-R0`: true-neighbour information effect; this is the primary causal contrast;
- `R1-B`: final deployable total effect.

No E-series, axial, attention, presence, weak-label, boundary-loss, spacing-aware, more-slice, or second-backbone variant is allowed.

## 2. Frozen B checkpoint

For each fold `f`, B is the existing multiclass 2D result:

```text
D:\BHSD_server_backups\multiclass_2d_min300_patience100\fold_f\checkpoint_best.pth
trainer = nnUNetTrainer_BHSDEarlyStop
dataset = Dataset001_BHSD
configuration = 2d
plans = nnUNetPlans
```

Before code implementation, record SHA-256 for all five checkpoints and verify that R0 and R1 for a fold read the identical file. B is not retrained. Existing B validation predictions are the baseline performance record; loading the checkpoint for the frozen forward pass does not authorize new saved-model inference in the present audit.

## 3. Center backbone freeze

The complete center model, including encoder, decoder, segmentation heads, normalization affine parameters, and all buffers, is frozen:

```text
center.eval()
for p in center.parameters():
    p.requires_grad_(False)
with torch.no_grad():
    z_B = center(x0)
```

The wrapper must keep the center in evaluation mode even when `wrapper.train()` is called. A before/after bytewise state-dict hash and parameter-gradient audit is mandatory. The optimizer may contain only parameters whose fully qualified names start with the residual-branch namespace.

Exactly one complete center backbone forward is permitted for B, R0, and R1. The lightweight slice stem is not counted as another complete backbone.

## 4. Locked residual architecture

This is a logit residual, not a feature residual. It is symmetric to neighbour exchange.

Each input slice is passed through the same shared shallow stem `phi`:

```text
Conv2d(1,16,3,padding=1,bias=True)
InstanceNorm2d(16,affine=True)
LeakyReLU
Conv2d(16,16,3,padding=1,bias=True)
InstanceNorm2d(16,affine=True)
LeakyReLU
```

Then:

```text
c = phi(x0)
n = (phi(x-1) + phi(x+1)) / 2
h = concat(c, n, stopgrad(z_B_fullres))       # 16+16+6 = 38 channels
h = Conv2d(38,32,3,padding=1) + IN + LeakyReLU
h = Conv2d(32,16,3,padding=1) + IN + LeakyReLU
delta_full = Conv2d(16,6,1)
```

The final `16->6` convolution weight and bias are exactly zero initialized. All earlier residual layers use one locked initialization rule shared by R0/R1. The expected residual-branch parameter count is 18,342 with affine instance normalization and convolution biases; the implementation test must calculate rather than trust this number.

If deep supervision is enabled, `delta_full` is bilinearly resized with `align_corners=False` to every frozen B logit scale and added to the corresponding B logits. Thus zero initialization restores every B deep-supervision tensor exactly. At inference deep supervision is disabled and the output is one `[N,6,H,W]` tensor.

Setting a non-trainable `delta_enabled=0` multiplier must restore B logits bit-for-bit. This switch is for identity tests only and cannot be tuned.

## 5. R0 and R1 input construction

The Stage3 loader first samples the real triplet and the center GT, applies one shared geometric transform to all three images and GT, and samples one shared intensity transform for the three image channels. In particular, blur, brightness, contrast, low-resolution simulation, and gamma must use synchronized channel parameters for Stage3.

After augmentation:

- R0 constructs its branch input as `[x0_aug,x0_aug,x0_aug]` and must prove exact tensor equality among the three channels;
- R1 uses `[x-1_aug,x0_aug,x+1_aug]`;
- the frozen center backbone receives only `x0_aug` in both arms;
- GT is only the transformed center mask.

R0 and R1 run the shared stem three times, even when R0 channels are identical, to keep the computational path and pass count matched. Swapping R1 neighbours must not change its output beyond the declared exact tolerance because neighbour features are averaged.

This Stage3-specific intensity rule is necessary: the current historical nnU-Net transform uses channel-independent intensity settings for most transforms, so duplication before that transform would not preserve R0 as duplicated-center.

## 6. Locked training policy

- Dataset/split: locked `Dataset001_BHSD/splits_final.json`, SHA-256 `A7F3088C3195273FEFFAA06A99E9A8F2C62F6AEB0AC5DC97A8498D1D5C55BEEA`.
- Patch/batch: `256x256`, batch 12.
- Loss: standard nnU-Net multiclass Dice + cross entropy with the standard deep-supervision weights.
- Optimizer: SGD over residual parameters only; initial LR `0.01`, momentum `0.99`, Nesterov true, weight decay `3e-5`.
- Schedule: polynomial nnU-Net schedule locked to maximum 1000 epochs.
- Stop/checkpoint: minimum 300 epochs; patience 100; minimum improvement `1e-4`; monitor online `ema_fg_dice`; report only full-case `checkpoint_best.pth` results; retain final checkpoint.
- Model seeds: `3407`, `1234`, `5678` for the fold-1 gate.
- Data seed: `1003410` for every arm/seed, reapplied after model initialization and by epoch.
- Augmentation workers: zero; single-thread only.
- Determinism: Python, NumPy, PyTorch, CUDA, and `PYTHONHASHSEED` set in the actual child process; deterministic algorithms requested with warnings saved.
- R0/R1 pairing: same checkpoint hash, residual initialization seed, sample order, crop, geometric transform, intensity transform, optimizer schedule, iterations, evaluation cases, and metric code within a seed. Only the predeclared neighbour tensor differs.

## 7. Fold use and no-interim-look rule

Fold 0 is closed for model selection. It may be used only for shape, forward/backward, identity, gradient, FLOP/parameter, memory/latency, and short loader smoke tests that do not save a trained checkpoint or compute fold-0 Dice.

Fold 1 is the predeclared development gate. All six R0/R1 jobs for seeds `3407/1234/5678` must have configs and hashes frozen together before any result is opened. No parameter or structure may change between jobs.

Folds 2-4 are confirmatory. They run only after the complete fold-1 gate passes, with the exact locked implementation and seed 3407. R0/R1 for all three folds are configured together; no fold result may be used to alter a later fold.

Seeds are repeated training realizations, not independent patients. Fold 1 and folds 2-4 must be reported separately.

## 8. Endpoints

Primary mechanism endpoint:

```text
delta_neighbor = class-balanced GT-present macro Dice(R1)
               - class-balanced GT-present macro Dice(R0)
```

Key total-effect endpoint:

```text
delta_center = class-balanced GT-present macro Dice(R1)
             - class-balanced GT-present macro Dice(B)
```

Capacity diagnostic: the corresponding `R0-B` delta.

Required secondary outcomes are nnU-Net `foreground_mean.Dice`, per-class GT-present Dice, pooled present patient-class Dice, absent-class any-FP and FP volume, lesion-level F1/recall, component-level small-lesion recall below 1mL, HD95 in mm, NSD@3mm, parameters, FLOPs, peak allocated GPU memory, batch-1 latency, full validation wall time, and complete backbone passes. Exact metric definitions and empty rules are frozen in `metric_definition_audit.csv`.

Because the primary endpoint uses only GT-present classes, a positive `delta_neighbor` cannot be produced merely by removing predictions for absent classes. The absent-FP table is still reported to show the separate error trade-off. A gain only in nnU-Net foreground mean with nonpositive present macro is a failure.

## 9. Fold-1 success and stop rule

R1 advances only if every condition holds after all six jobs finish:

1. `R1-B` present-macro delta is positive for all three seeds;
2. mean three-seed `R1-B` present-macro delta is at least `+0.015`;
3. `R1-R0` present-macro delta is positive for at least two of three seeds;
4. mean three-seed `R1-R0` present-macro delta is at least `+0.010`;
5. at least three of five classes have a positive mean `R1-R0` present-Dice delta;
6. no class has mean `R1-R0` present-Dice delta below `-0.020`;
7. R1 does not obtain its apparent benefit only from absent-class FP: conditions 1-6 use GT-present support and must pass independently of the absent-FP direction;
8. R0 and R1 each use one complete center backbone pass;
9. R1 batch-1 median inference latency is no more than `1.25x` B under the same warm-up/device procedure;
10. R1 peak allocated memory is no more than `1.30x` B;
11. R0/R1 parameter counts and complete-backbone pass counts are exactly equal;
12. all required implementation tests pass and no result/case/checkpoint provenance check fails.

Failure of any item stops Stage3 after fold 1. No R2, gate, presence head, attention, loss change, or extra seed is allowed.

These thresholds are intentionally larger than trivial numerical noise: the historical E0/E2 three-seed fold0 present-macro deltas were `-0.0301`, `+0.0513`, and `+0.0168`, demonstrating large initialization heterogeneity, while each historical controlled 2.5D run required roughly 7-10 V100 hours. Six development jobs are therefore the maximum justified gate before cross-fold confirmation.

## 10. Confirmatory rule on folds 2-4

For each validation patient, compute paired R1-R0 and R1-B outcomes. Pool only the disjoint folds 2-4 for confirmation. Use 10,000 patient-cluster bootstrap samples with RNG seed `20260726`; when a patient is sampled, all of that patient's classes and all compared models move together. Do not use seed as a patient cluster and do not include fold1 in the confirmatory confidence interval.

Confirmation succeeds only if all conditions hold:

1. pooled class-balanced present `R1-R0 >= +0.010` and its patient-bootstrap 95% percentile CI lower bound is above zero;
2. at least two of the three fold-specific R1-R0 deltas are positive;
3. pooled class-balanced present `R1-B >= +0.015` and its 95% CI lower bound is above zero;
4. at least three of five pooled class-specific R1-R0 present-Dice deltas are positive;
5. no pooled class-specific R1-R0 present-Dice delta is below `-0.020`;
6. all compute constraints from the fold-1 gate remain satisfied.

The bootstrap quantifies patient uncertainty conditional on these fixed folds. It does not create an independent test set and does not estimate uncertainty over alternative fold assignments.

If confirmation fails, the result is a controlled negative/heterogeneous result and all new 2.5D module development stops. If it succeeds, the publication claim remains limited to the locked BHSD cross-validation protocol unless an independent test set is later obtained.

## 11. Freeze package required before any training approval

The implementation phase must create a new isolated directory containing:

- exact B checkpoint hashes for folds 1-4;
- Stage3 model/loader/evaluator code hashes;
- R0/R1 configs and seed table;
- exact train/validation case lists;
- parameter/FLOP/pass-count report;
- all required synthetic test logs;
- Git commit hash (after a separate approval to commit);
- a no-interim-look declaration.

No Stage3 job may start until that package is complete and separately approved.

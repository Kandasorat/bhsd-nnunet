# BHSD nnU-Net / Attention Project Handoff

> **Authoritative snapshot: 2026-07-22**
>
> This file is the current source of truth for continuing the project. Obsolete
> Google Cloud instructions and superseded Gadi snapshots have been removed.

## New-chat bootstrap

```text
Please read C:\Users\92127\OneDrive - UNSW\project_linpeng\PROJECT_HANDOFF.md
completely and continue the NCI Gadi BHSD project from "Immediate next work".
Do not rerun completed baselines or the completed standard 2.5D fold-0 runs.
Do not describe harmonized attention adaptations as official reproductions.
```

Canonical file:

```text
C:\Users\92127\OneDrive - UNSW\project_linpeng\PROJECT_HANDOFF.md
```

Repository copy:

```text
C:\Users\92127\OneDrive - UNSW\project_linpeng\code\docs\PROJECT_HANDOFF.md
```

## Project scope

Run reproducible BHSD intracranial-haemorrhage segmentation experiments on NCI
Gadi. Keep these task definitions separate:

1. `Dataset001_BHSD`: background + EDH, IPH, IVH, SAH, SDH.
2. `Dataset002_BHSD_Binary`: background versus the union of labels 1-5.

Completed formal five-fold baselines:

- multiclass nnU-Net 2D and 3D full resolution;
- binary nnU-Net 2D and 3D full resolution.

All 20 PBS subjobs finished with `Exit_status = 0`. Complete result trees were
downloaded and verified locally, then the four server result directories were
deleted. Do not rerun them without a new documented experimental reason.

Completed standard three-slice 2.5D fold-0 runs under the locked harmonized
policy:

- multiclass `Dataset001_BHSD`, job `174332417`, `Exit_status = 0`;
- binary `Dataset002_BHSD_Binary`, job `174332422`, `Exit_status = 0`.

Their complete fold-0 result trees have been downloaded locally. Their server
copies are intentionally retained until the current A1-A8 screen is complete
and all server/local file counts have been compared.

## Gadi and environment

- User/project: `ly6399` / `ke17`.
- Login: `ssh ly6399@gadi.nci.org.au`.
- Transfer host: `gadi-dm.nci.org.au`.
- Root: `/scratch/ke17/bhsd-nnunet`.
- Git checkout: `/scratch/ke17/bhsd-nnunet/software/bhsd-nnunet`.
- GPU queue/hardware: `gpuvolta`, Tesla V100-SXM2-32GB.
- Python module: `python3/3.10.4`.
- Environment: `/scratch/ke17/bhsd-nnunet/envs/bhsd-nnunet-py310`.
- PyTorch/nnU-Net: `2.7.1+cu118` / `2.6.4`; CUDA runtime 11.8.

Activation:

```bash
export BHSD_ROOT=/scratch/ke17/bhsd-nnunet
export nnUNet_raw="$BHSD_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$BHSD_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$BHSD_ROOT/runs/nnUNet_results"

module purge
module load python3/3.10.4
source "$BHSD_ROOT/envs/bhsd-nnunet-py310/bin/activate"
```

Dataset001 and Dataset002 each use the same five-fold split over 192 cases.
Current plans use a 256 x 256 patch for 2D/2.5D and 28 x 256 x 256 for 3D.
A 512 x 512 full-validation image shape does not change the training patch.

## Locked harmonized nnU-Net policy

- Maximum epochs: 1000.
- Minimum epochs: 300.
- Patience: 100 after the minimum-duration condition.
- Minimum improvement: 0.0001.
- Monitor: `ema_fg_dice`.
- Primary checkpoint: `checkpoint_best.pth`.
- Full-case validation: `--val_best`.
- Probability export: `--npz`.
- Retain `checkpoint_final.pth` for audit.

This policy applies to harmonized nnU-Net comparisons. Source-faithful CSAM and
CSA-Net experiments must first follow their released paper/code protocols;
protocol deviations must be declared rather than silently harmonized.

## Formal baseline results

| Task | PBS array | Stop epochs, folds 0-4 | Dice, mean +/- sample SD |
|---|---|---|---:|
| Multiclass 2D | `174241149[]` | 445, 553, 582, 525, 399 | 0.270792 +/- 0.015635 |
| Multiclass 3D | `174241150[]` | 976, 721, 561, 520, 448 | 0.306509 +/- 0.034337 |
| Binary 2D | `174241151[]` | 464, 541, 475, 449, 414 | 0.598194 +/- 0.030126 |
| Binary 3D | `174241152[]` | 399, 506, 505, 667, 416 | 0.587289 +/- 0.035918 |

Per-fold foreground Dice:

| Fold | Multi 2D | Multi 3D | Binary 2D | Binary 3D |
|---:|---:|---:|---:|---:|
| 0 | 0.269076 | 0.356454 | 0.640113 | 0.617389 |
| 1 | 0.264326 | 0.294809 | 0.596216 | 0.546341 |
| 2 | 0.288631 | 0.308782 | 0.557025 | 0.564384 |
| 3 | 0.282818 | 0.311337 | 0.608122 | 0.631400 |
| 4 | 0.249109 | 0.261163 | 0.589491 | 0.576932 |

Multiclass five-fold class means:

| Class | 2D | 3D |
|---|---:|---:|
| EDH | 0.063942 | 0.106299 |
| IPH | 0.503643 | 0.554609 |
| IVH | 0.468428 | 0.489870 |
| SAH | 0.182184 | 0.217185 |
| SDH | 0.135762 | 0.164582 |

Interpretation:

- multiclass 3D improves every subtype and exceeds 2D by about 0.0357 macro
  Dice;
- EDH, SAH, and SDH remain the principal multiclass weaknesses;
- binary 2D is about 0.0109 above binary 3D, but five folds alone do not prove
  statistical superiority;
- binary and multiclass Dice are different endpoints;
- all reported values are cross-validation validation results, not an
  independent test-set evaluation.

## Standard three-slice 2.5D fold-0 results

These are the A0 references for the current harmonized module screen. Model
selection used online foreground-Dice EMA, but every reported number below is
from full-case `validation/summary.json` using `checkpoint_best.pth`. Online EMA
values must not be compared directly with summary Dice.

| Task | Job | Stop epoch | Walltime | Foreground Dice |
|---|---|---:|---:|---:|
| Multiclass 2.5D A0 | `174332417` | 411 | 02:22:35 | 0.253109 |
| Binary 2.5D A0 | `174332422` | 568 | 03:01:57 | 0.632198 |

Multiclass A0 class Dice:

| Class | 2.5D A0 | 2D fold 0 | 3D fold 0 |
|---|---:|---:|---:|
| EDH | 0.120320 | 0.100987 | 0.225921 |
| IPH | 0.558220 | 0.526673 | 0.622791 |
| IVH | 0.351645 | 0.450625 | 0.492130 |
| SAH | 0.125985 | 0.111412 | 0.206173 |
| SDH | 0.109375 | 0.155683 | 0.235254 |

Interpretation is limited to fold 0: multiclass A0 is 0.015967 below 2D and
0.103345 below 3D; binary A0 is 0.007915 below 2D and 0.014809 above 3D. The
three-slice input therefore did not automatically improve multiclass subtype
separation. A1-A8 test whether the fusion mechanism, rather than the presence
of neighbouring slices alone, changes that result.

## Completed multiclass A1-A8 screen

PBS array `174338292[]` was submitted from Git revision `1167efe`. All eight
V100 subjobs completed with `Exit_status = 0`, and all result trees were
downloaded locally and checked for best/final checkpoints, logs, debug files,
progress plots, and full-case validation summaries.

| Array index | ID | Trainer mechanism |
|---:|---|---|
| 0 | A1 | common adapter capacity control, no attention |
| 1 | A2 | CSAM Slice Gate |
| 2 | A3 | ECA Slice Gate |
| 3 | A4 | XAG-Net-style pixel-wise cross-slice attention |
| 4 | A5 | CSA-Net-style centre-to-neighbour attention |
| 5 | A6 | CBAM |
| 6 | A7 | Coordinate Attention |
| 7 | A8 | axial/P3D-style slice convolution control |

All arms are multiclass `Dataset001_BHSD` fold 0. They share the same standard
nnU-Net 2D backbone, three-slice input, split, augmentation, Dice + CE loss,
optimizer/schedule, early stopping, checkpoint rule, and final nnU-Net summary
evaluation. The module code is class-count agnostic, but binary A1-A8 jobs have
not been submitted. Do not submit the superseded
`train_lightweight_slice_attention_25d_fold0.pbs`.

| Rank | Arm | Foreground Dice | Stop epoch | Total hours |
|---:|---|---:|---:|---:|
| 1 | A8 axial slice convolution | 0.295321 | 730 | 4.5185 |
| 2 | A5 CSA centre-to-neighbour | 0.272406 | 432 | 2.7589 |
| 3 | A2 CSAM slice gate | 0.265697 | 580 | 3.3498 |
| 4 | A1 adapter control | 0.263102 | 475 | 2.7656 |
| 5 | A4 pixel-wise cross-slice | 0.262461 | 598 | 3.4757 |
| 6 | A7 coordinate attention | 0.257809 | 399 | 2.4143 |
| 7 | A6 CBAM | 0.252767 | 411 | 2.5643 |
| 8 | A3 ECA slice gate | 0.249636 | 422 | 2.4355 |

A8 exceeded A0 by 0.042212 and 2D fold 0 by 0.026245, but remained 0.061133
below 3D fold 0. A5 was the strongest attention arm. The old YAML seed was set
only in the parent runner while `nnUNetv2_train` ran in a child process, and
the augmentation workers used `seeds=None`; therefore small A1-A8 differences
are not strict same-seed paired evidence. The results remain valid for
screening A8 and A5 into the controlled fusion experiment.

## Revised slice-spectrum D0-D6 screen (code ready, not yet run)

The earlier difference-based proposal was revised after prior-art review. Slice
differencing itself is not new and must not be claimed as such: SDC-UNet already
applies Slice Difference Convolution to intracranial-haemorrhage CT segmentation
(doi:10.15938/j.jhust.2025.03.001), with related adjacent-slice difference or
interaction ideas in CLDSINet, ASD-Net, and ACSFormer.

The implemented experiment is therefore a controlled, falsifiable ablation of
the orthonormal graph-Fourier basis of a three-node slice path:

```text
Z0 = (F- + F0 + F+) / sqrt(3)       low-frequency persistence
Z1 = (F- - F+) / sqrt(2)            signed orientation contrast
Z2 = (F- - 2 F0 + F+) / sqrt(6)     even curvature contrast
```

These are slice-index spectral contrasts, not physical z derivatives. D0 is a
centre-descriptor capacity control; D1-D3 isolate Z0, Z1, and gated Z2; D4 uses
all bands; D5 learns pixel-wise softmax band weights while retaining direction;
D6 tests hard neighbour-swap invariance by averaging complete predictions for
the original and reversed-neighbour inputs. D6 deliberately uses two backbone
passes and must be judged on runtime as well as Dice.

All seven multiclass fold-0 configs use isolated trainer/output namespaces,
model seed 3407, data seed 1003410, epoch-indexed train/validation seeds,
single-thread augmentation, best-effort deterministic CUDA, and the locked
nnU-Net checkpoint/evaluation policy. Verification covers orthonormal energy,
neighbor-reversal parity, controlled adapter capacities, zero-initialized
residual identity, deep supervision, backward gradients, and exact D6 swap
invariance. A real Dataset001 plans initialization audit confirmed all seven
trainers use three input channels, batch 12, six output heads, 256 x 256
patches, and standard nnU-Net SGD with initial LR 0.01. See
`docs/SPECTRAL_SLICE_SCREEN.md`. No D0-D6 Gadi job has been submitted at this
snapshot.

Compute cost is a co-primary screening constraint, not a proxy for quality.
Every new C0-C3/F1-F2 and D0-D6 subjob profiles total/adapter parameters,
batch-1 forward latency, allocated GPU-memory peak, and backbone passes before
training; the runner separately records complete wall time and sampled training
memory. `scripts/summarize_controlled_screens.py` combines these measurements
with full-case Dice, ratios to C0/D0, per-class regression, and Pareto status.
F1/F2 candidacy requires at least +0.01 foreground Dice over C3; spectral arms
use D0 as the automatic reference, with mechanism-specific comparisons also
reported. All candidates require no class decrease worse than 0.02 and no
domination by an equally accurate lower-cost arm. D6 is explicitly a
two-backbone-pass experiment.

## Verified local backups

Each current backup contains 609 files, including five each of
`checkpoint_best.pth`, `checkpoint_final.pth`, `summary.json`, `progress.png`,
`debug.json`, and training logs. No files are OneDrive offline placeholders.

Base directory:

```text
C:\Users\92127\OneDrive - UNSW\project_linpeng\server_backups
```

| Backup directory | Local file bytes |
|---|---:|
| `multiclass_2d_min300_patience100` | 34,428,342,596 |
| `multiclass_3d_min300_patience100` | 30,955,821,300 |
| `binary_2d_min300_patience100` | 6,065,512,354 |
| `binary_3d_min300_patience100` | 6,009,003,037 |
| `multiclass_25d_3slice_fold0_min300_patience100` | 7,199,232,516 |
| `binary_25d_3slice_fold0_min300_patience100` | 1,264,183,758 |

The complete A1-A8 result trees are under
`multiclass_25d_attention_screen_partial_2026-07-22`; each arm contains 126
files and the expected best/final checkpoints and validation summary.

Server `du -sb` was 86,016 bytes larger per tree because it included directory
metadata. This is not missing content.

Older patience-50 provenance backups remain as
`2d_baseline_2026-07-20` and `3d_baseline_full_2026-07-20`. The four formal
min-300/patience-100 runs above are the current baselines.

Each 2.5D fold-0 backup currently contains 123 files and includes
`checkpoint_best.pth`, `checkpoint_final.pth`, `debug.json`, `progress.png`, a
training log, and `validation/summary.json`. Before deleting either Gadi copy,
compare the server file count and summed file bytes with the local values.

## Current server-result state

The four formal result trees under `runs/nnUNet_results` were removed only
after local verification. Small Dataset001/Dataset002 parent directories may
remain. Live post-cleanup usage was approximately:

```text
runs 1.2M; cache 3.0G; envs 6.3G; data 13G; whole BHSD root about 22G
data: nnUNet_raw 2.8G; archives 4.2G; nnUNet_preprocessed 6.0G
```

Do not delete raw/preprocessed data, environment, or cache before upcoming
experiments. NCI storage reports can lag behind live `du` after deletion.

The completed multiclass/binary A0 and A1-A8 server trees may still exist under
`runs/nnUNet_results`. Their verified local copies are the preservation source;
do not delete any new C0-C3/F1-F2 tree before downloading and checking it.

Last compute report: grant 200.00 KSU, used 5.36 KSU, reserved 0.00 KSU,
available 194.64 KSU. The completed baseline GPUs have been released.

## Git and attention-method boundary

- GitHub: `https://github.com/Kandasorat/bhsd-nnunet.git`.
- Branch: `main`.
- Last pushed fusion-screen implementation revision: `fcdda53`.
- Gadi A1-A8 array `174338292[]` was launched from revision `1167efe`.
- Revised and locally verified D0-D6 slice-spectrum implementation revision:
  `4fa9572`.
- Full-chain post-implementation audit and strengthened verification revision:
  `6550ff3`.

Read `docs/ATTENTION_REPRODUCTION_POLICY.md` before attention work. Current
CSAM/CSA-Net configs are labelled `harmonized_nnunet_adaptation` and
`source_faithful: false`; do not report them as official reproductions.

Upstream protocol anchors from the completed audit:

- CSAM code defaults: 150 epochs, 20-slice sequence, input 128, Adam 1e-4,
  cross entropy, batch 2, base width 64. Its released repository is
  incomplete/non-turnkey, so a BHSD port needs explicit documentation.
- CSA-Net: previous/centre/next slices, R50-ViT-B16 ImageNet21k initialization,
  input 224, 40 epochs, batch 16, SGD 0.001, momentum 0.9, weight decay 1e-4,
  0.5 CE + 0.5 Dice, poly LR, validation every five epochs after epoch 10,
  seed 1234.

Architecture fidelity and training-protocol fidelity must be reported
separately.

The A1-A8 screen is a harmonized nnU-Net module comparison, not eight official
end-to-end reproductions. Pinned upstream provenance, preserved equations, and
necessary deviations are recorded in `docs/ATTENTION_MODULE_SCREEN.md`.

## Immediate next work

1. On Gadi, pull `main`, confirm the displayed revision matches the latest
   pushed commit, activate the existing environment, and run
   `scripts/check_gadi_ready.py --server --require-binary` plus
   `scripts/verify_spectral_slice_fusion.py`.
2. Submit `hpc/gadi/train_25d_fusion_screen_fold0.pbs` once if it has not yet
   been submitted. Its six indices are C0-C3 and F1-F2. Do not mix their new
   controlled namespaces with A0-A8 paths.
3. The D0-D6 spectral screen is independently runnable with
   `hpc/gadi/train_25d_spectral_screen_fold0.pbs`. It is a seven-job multiclass
   fold-0 ablation, not an official reproduction or a novelty claim. Record the
   PBS job ID before leaving the shell.
4. Require `Exit_status = 0`, inspect warnings/errors, then download each
   completed array's result trees together. Rank only the best-checkpoint
   full-case `validation/summary.json`; retain timing and `compute_profile.json`.
   Run `scripts/summarize_controlled_screens.py` for the relevant screen rather
   than ranking by Dice alone. D6's two-pass runtime must be charged in full.
5. Advance a fusion only if it materially exceeds controlled C3 (target about
   +0.01 foreground Dice) without a major per-class regression. Advance a
   spectral arm only if it beats D0 and the appropriate ungated control with a
   coherent subtype pattern. Fold 0 remains screening evidence.
6. Confirm any winner and necessary controls over multiple seeds and then
   multiclass folds 1-4. Binary confirmation comes later; do not run all
   discarded arms on binary.
7. CUDA loss/pooling kernels in this PyTorch stack are not guaranteed bitwise
   deterministic; report these as controlled/best-effort reproducible and use
   multi-seed confirmation for claims.

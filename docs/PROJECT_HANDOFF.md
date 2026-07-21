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

## Active multiclass A1-A8 screen

PBS array `174338292[]` was submitted from Git revision `1167efe`. At this
snapshot all eight V100 subjobs are in state `R`; the most recent observed
elapsed walltime was about 02:23. Query with:

```bash
qstat -x -t "174338292[].gadi-pbs"
```

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

Timing is not lost for this array. The runner writes total duration and sampled
GPU utilization/memory to
`$BHSD_RESULTS_DIR/<experiment_name>/stage_metrics.csv` and
`train_fold_0_resource_samples.csv`; PBS history separately provides
`resources_used.walltime`. Because this array was launched before the later
fold-local timing enhancement, download its `experiment_metadata` directories
as well as the nnU-Net result trees. Future runs additionally write
`fold_*/run_timing.json` beside their checkpoints.

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

The completed multiclass/binary 2.5D A0 result trees and all live A1-A8 result
trees remain under `runs/nnUNet_results`. Do not delete them while the screen is
running. After A1-A8 complete, download all completed experiment trees with one
local PowerShell batch script, verify key artifacts plus server/local counts and
bytes, and only then consider server cleanup.

Last compute report: grant 200.00 KSU, used 5.36 KSU, reserved 0.00 KSU,
available 194.64 KSU. The completed baseline GPUs have been released.

## Git and attention-method boundary

- GitHub: `https://github.com/Kandasorat/bhsd-nnunet.git`.
- Branch: `main`.
- Last verified and pushed code revision: `1167efe`.
- Gadi array `174338292[]` was launched from that revision.
- Canonical and repository-copy handoff files were synchronized on 2026-07-22;
  this documentation update does not change the code used by active jobs.

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

1. Monitor `174338292[]`; do not resubmit A0 or any A1-A8 arm while its current
   job/result path exists.
2. When the array finishes, require `Exit_status = 0` for every index 0-7 and
   scan PBS/training logs for tracebacks, OOM, killed processes, or incomplete
   validation.
3. Run `scripts/summarize_attention_screen.py`. Compare only full-case
   `validation/summary.json` foreground/class Dice; A0 is 0.253109. Do not rank
   methods by online EMA.
4. Use one local PowerShell batch script to download all eight complete result
   trees into distinctly named directories under `server_backups`. Verify
   `checkpoint_best.pth`, `checkpoint_final.pth`, `summary.json`, file counts,
   and summed file bytes against Gadi.
5. Build the A0-A8 fold-0 table and inspect per-class effects, especially EDH,
   IVH, SAH, and SDH. Treat fold-0 ranking as screening evidence only.
6. Select the most defensible one or two mechanisms using a frozen rule that
   considers Dice, class balance, stability, parameters, memory, and runtime.
   Then decide whether to run confirmatory multiclass folds and later binary
   confirmation. Do not run all binary A1-A8 arms by default.
7. Preserve the distinction between source-faithful CSAM/CSA-Net experiments
   and these harmonized module adaptations in all tables and writing.

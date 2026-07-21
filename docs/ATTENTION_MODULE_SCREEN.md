# Harmonized 2.5D Attention Module Screen

## Scope

The implementation is output-class agnostic, but the first screen is locked to
`Dataset001_BHSD` multiclass fold 0. A0 is the already running standard
three-slice 2.5D baseline. A1-A8 use the same nnU-Net 2D backbone, data split,
three-slice input, 256x256 patch, batch sampling, augmentation, Dice + CE loss,
SGD/poly schedule, online foreground-Dice EMA stopping policy, and
`checkpoint_best.pth` validation.

All eight arms share an 8-channel per-slice spatial descriptor and a residual
1x1 projection into the original stacked input. The projection is zero
initialized, so every arm starts exactly from A0's input. A1 is essential: it
tests whether any improvement is caused merely by the added descriptor and
projection rather than by an attention mechanism.

| ID | Trainer | Mechanism | What is retained from the source |
|---|---|---|---|
| A1 | `nnUNetTrainer_25D_AdapterControl` | no attention | common adapter capacity only |
| A2 | `nnUNetTrainer_25D_CSAMSliceGate` | CSAM slice gate | average/max slice descriptors, shared 4x expansion MLP, sigmoid multiplication |
| A3 | `nnUNetTrainer_25D_ECASliceGate` | ECA slice gate | GAP, non-reducing 1D convolution, sigmoid multiplication |
| A4 | `nnUNetTrainer_25D_PixelWiseCrossSlice` | pixel-wise CSA | spatially varying slice scores and softmax across adjacent slices |
| A5 | `nnUNetTrainer_25D_CSACenterNeighbor` | CSA-Net cross attention | theta/phi/g non-local attention for previous, self, and next branches |
| A6 | `nnUNetTrainer_25D_CBAM` | CBAM | average/max channel MLP followed by a 7x7 spatial gate |
| A7 | `nnUNetTrainer_25D_CoordinateAttention` | Coordinate Attention | horizontal/vertical pooling, shared bottleneck and direction-specific gates |
| A8 | `nnUNetTrainer_25D_AxialSliceConv` | axial convolution | residual 3x1x1 slice-axis convolution as a non-attention control |

## Provenance and deviations

- CSAM: `aL3x-O-o-Hung/CSAM` at
  `a0029206ef3b4147351813b7d67eb7b5964c8f33`. The official SliceAttentionModule
  reduces over channel and space, applies a shared slice MLP to average and max
  descriptors, sums them, and uses sigmoid weights. The screen makes this
  operation batched and deterministic; the official stochastic low-rank
  uncertainty branch is excluded so A2 tests the slice gate alone.
- ECA-Net: `BangguWu/ECANet` at
  `b332f6b3e6e2afe8a3287dc8ee8440a0fbec74c4`. The original channel axis is
  reinterpreted as the ordered slice axis. Kernel size 3 is fixed because the
  experiment has exactly three adjacent slices.
- Pixel-wise CSA: XAG-Net, arXiv `2508.06258v1`. The paper's defining operation
  is reimplemented exactly at the module level as
  `X + X * softmax(Conv1x1(X), slice_axis)`: slice weights vary at every pixel
  and are normalized over adjacent slices. No author code repository was identified, so this arm
  must be described as paper-derived rather than source-code faithful.
- CSA-Net: `mirthAI/CSA-Net` at
  `9be2dbe8d2247ab91d03f18bd8af92448a675ff9`. Its theta/phi/g matrix rule and
  three previous/self/next branches are retained. The official file creates 16
  attention blocks inside `forward`, which makes their parameters unregistered;
  the adaptation registers its branches in `__init__`. Attention runs on a
  16x16 descriptor grid instead of 1024-channel full features to fit the common
  nnU-Net adapter budget. The source block's zero-initialized output BN is not
  duplicated: the common residual projection already supplies exact identity
  initialization, while a second zero initialization before ReLU would leave
  the adapted attention branch with zero gradient.
- CBAM: `Jongchan/attention-module` at
  `459efad0e05ee7dde50c41ca10a3d0800bc3792a`. Its channel-then-spatial ordering
  and pooling rules are retained; the input channels are the concatenated
  descriptors of three slices.
- Coordinate Attention: `houqb/CoordAttention` at
  `7619bea9acbe260b3793833cc78cef3f124c8112`. Its two directional pooling paths,
  hard-swish bottleneck, and two sigmoid gates are retained; slice descriptors
  are concatenated along the channel axis.
- P3D: `ZhaofanQiu/pseudo-3d-residual-networks` at
  `e91f95de5ae7e69183ea014e06ac105debb0cfe9`. A8 is only a P3D-inspired
  temporal convolution control, not a reproduction of the full P3D ResNet.

These are harmonized adaptations, not claims of source-faithful end-to-end
reproduction. The paper algorithms are preserved where compatible, while the
backbone and training protocol remain nnU-Net so the comparison isolates the
fusion mechanism.

## Gadi run

After the current A0 jobs finish and the repository is pulled:

```bash
cd /scratch/ke17/bhsd-nnunet/software/bhsd-nnunet
git pull --ff-only
python scripts/check_gadi_ready.py
qsub hpc/gadi/train_25d_attention_screen_fold0.pbs
```

The single `qsub` creates array indices 0-7 for A1-A8. Do not submit the older
`train_lightweight_slice_attention_25d_fold0.pbs` in addition to A2; it is a
superseded pilot with a different adapter capacity.

## Timing and resource records

`scripts/run_experiment.py` records runner start/end UTC timestamps, total
duration, exit code, sampled GPU utilization/memory, and PBS identifiers. The
central records are written to:

```text
$BHSD_RESULTS_DIR/<experiment_name>/stage_metrics.csv
$BHSD_RESULTS_DIR/<experiment_name>/train_fold_0_resource_samples.csv
```

For runs launched after the timing-record enhancement, the same training
summary is also copied to `fold_0/run_timing.json`, so a normal result-tree
download includes it. `duration_seconds` covers the nnU-Net training command
and its post-training full validation. PBS `resources_used.walltime` remains the
authoritative scheduler walltime and may be slightly longer because it also
includes environment setup and extension installation.

# D0-D6 controlled slice-spectrum screen

## Research boundary

This experiment does **not** claim that slice differencing is new. The closest
known prior is SDC-UNet, which applies Slice Difference Convolution to
intracranial-haemorrhage CT segmentation (Qiu, Xie, and Sha, 2025,
doi:10.15938/j.jhust.2025.03.001). CLDSINet combines static features,
inter-slice difference images, and dynamic weighting; ASD-Net disentangles
similar and independent adjacent-slice features; ACSFormer performs adaptive
token-specific inter-slice interaction and reports results on INSTANCE ICH.

The purpose here is narrower and falsifiable: under one locked nnU-Net policy,
measure the independent contributions of the three orthogonal modes of a
three-slice path and determine whether orientation-sensitive or enforced
neighbor-swap-invariant use is more appropriate for multiclass BHSD.

Primary prior-art links:

- SDC-UNet: https://doi.org/10.15938/j.jhust.2025.03.001
- CLDSINet: https://doi.org/10.1016/j.inffus.2025.103509
- ASD-Net: https://doi.org/10.1016/j.bspc.2025.107809
- ACSFormer: https://www.sciencedirect.com/science/article/pii/S0031320325015584
- Central Difference Convolution: https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.html
- Temporal Difference Networks: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html
- Deep Sets: https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html

## Fixed mathematical representation

For encoded previous, centre, and next slice features
`F = [F-, F0, F+]`, use the orthonormal eigenbasis of the three-node path graph:

```text
Z0 = (F- + F0 + F+) / sqrt(3)       low-frequency persistence
Z1 = (F- - F+) / sqrt(2)            odd, orientation-sensitive contrast
Z2 = (F- - 2 F0 + F+) / sqrt(6)     even curvature contrast
```

The transform is orthonormal, so `||F||^2 = ||Z0||^2 + ||Z1||^2 + ||Z2||^2`.
It is a discrete slice-index spectrum. It must not be described as a physical
z derivative because this version does not divide by acquisition spacing.

All arms encode each slice with the same 8-channel descriptor, place their
selected modes into the same 24-channel context tensor, project it back to the
centre input channel only, and start with a zero-initialized projection. D0-D5
therefore begin as the same ordinary stacked three-slice input. D6 is a group
average of original and neighbor-swapped predictions, giving exact output
invariance at the cost of two backbone evaluations.

## Frozen arms

| ID | Context | Scientific question |
|---|---|---|
| D0 | centre descriptor in slot 0 | dormant-parameter/capacity control |
| D1 | `Z0` only | does low-pass persistence help? |
| D2 | signed `Z1` only | is superior/inferior-oriented variation useful? |
| D3 | `sigmoid(g(F0,abs(Z2))) * Z2` | does bounded curvature evidence help? |
| D4 | concatenated `Z0,Z1,Z2` | does the complete ungated spectrum help? |
| D5 | pixel-softmax weighted `Z0,Z1,Z2` | does adaptive oriented selection help? |
| D6 | weighted `Z0,abs(Z1),Z2` plus group averaging | is hard neighbor-swap invariance beneficial? |

D3 is gated because second differences amplify high-frequency noise. D5/D6
use a three-way softmax, preventing unbounded band weights. D6 is an ablation,
not an assumed improvement: axial CT order contains anatomical direction, so
hard invariance may discard useful information.

## Locked training/evaluation

- Dataset001_BHSD multiclass, fold 0, three slices, 256 x 256 patches.
- Model seed 3407 propagated into the actual training child process; a separate
  fixed data seed is reapplied after initialization so parameter-count
  differences cannot shift augmentation randomness. Train and validation RNGs
  are then reset from deterministic epoch-indexed seeds, including on resume.
- Single-thread augmentation and best-effort deterministic CUDA policy.
- nnU-Net optimizer/schedule and Dice + CE.
- Max 1000 epochs, min 300, EMA foreground-Dice patience 100.
- Select `checkpoint_best.pth`; report only full-case
  `validation/summary.json` Dice.
- Record total runtime and note that D6 intentionally performs two backbone
  forwards.

The screen tests plausibility, not publication novelty. Any winning D5/D6 arm
must be compared structurally with the full SDC-UNet paper and confirmed across
multiple seeds and folds before a new-method claim.

## Compute-cost rule

Each array arm profiles batch-1 inference before training and stores
`compute_profile.json` both with metadata and in the normal fold result tree.
The profile records total/adapter parameters, forward latency, allocated-memory
peak, and backbone passes per prediction. Actual training wall time and sampled
GPU memory are recorded separately by the experiment runner. D6 is expected to
report two backbone passes and is not preferred unless its gain justifies that
cost.

After D0-D6 complete, `scripts/summarize_controlled_screens.py --screen
spectral` combines Dice and compute measurements, reports ratios to D0, and
marks Pareto-dominated arms. Automatic candidacy requires at least +0.01
foreground Dice over D0, no class loss worse than 0.02, and Pareto efficiency;
mechanism-specific comparisons such as D5 versus D4 are still interpreted
manually.

## Verification and Gadi submission

The verifier checks transform energy and neighbor-reversal parity, controlled
adapter parameter counts, zero-initialized identity, deep-supervision output
structure, backward gradients, and exact D6 prediction invariance. A separate
real Dataset001 plans initialization audit confirmed three input channels,
batch 12, six segmentation heads, 256 x 256 patches, and the standard nnU-Net
SGD optimizer with initial learning rate 0.01 for every arm.

```bash
python3 scripts/verify_spectral_slice_fusion.py
qsub hpc/gadi/train_25d_spectral_screen_fold0.pbs
```

Array indices 0-6 map directly to D0-D6.

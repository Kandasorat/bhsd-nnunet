# Volume-wise CSAM source and adaptation

The CSAM modules and U-Net are based on the official MIT-licensed repository:

- https://github.com/aL3x-O-o-Hung/CSAM
- pinned upstream commit: `a0029206ef3b4147351813b7d67eb7b5964c8f33`
- paper: WACV 2024, *CSAM: A 2.5D Cross-Slice Attention Module for
  Anisotropic Volumetric Medical Image Segmentation*

BHSD adaptation:

- one ordered sequence is one network sample; unrelated patients are never
  mixed in the slice dimension;
- because the official slice-attention MLP requires a fixed sequence length,
  training uses contiguous 32-slice windows and full-case validation uses
  overlapping 32-slice windows to cover every slice;
- every window uses a common 256x256 spatial patch across all slices;
- uncertainty is sampled during training and its mean is used during eval so
  early stopping and `checkpoint_best.pth` selection are reproducible.

These changes adapt input handling and evaluation; semantic, positional,
slice, and uncertainty attention remain the official CSAM formulation.

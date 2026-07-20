# CSA-Net vendored source

Source: https://github.com/mirthAI/CSA-Net

Pinned upstream commit: `9be2dbe8d2247ab91d03f18bd8af92448a675ff9`

License: MIT; see `LICENSE` in this directory.

The following compatibility corrections are intentionally applied:

- the 16 cross-attention heads are registered in `__init__` instead of being
  newly allocated inside every `forward` call;
- the hard-coded `.cuda()` call is removed so PyTorch controls device
  placement;
- a small local `ConfigDict` replaces the optional `ml_collections`
  dependency. It does not change model configuration values.

Architecture, number of attention heads, ResNetV2/ViT encoder, decoder, and
pretrained-weight loading remain based on the upstream implementation.

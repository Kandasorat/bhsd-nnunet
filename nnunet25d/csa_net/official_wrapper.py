from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from nnunet25d.csa_net.vendor.vit_seg_modeling import CONFIGS, VisionTransformer


class OfficialCSANet3SliceWrapper(nn.Module):
    """nnU-Net adapter around the official three-slice CSA-Net architecture."""

    def __init__(
        self,
        input_channels_per_slice: int,
        num_classes: int,
        image_size: int = 256,
        pretrained_path: str | Path | None = None,
    ):
        super().__init__()
        if input_channels_per_slice != 1:
            raise ValueError(
                "The official CSA-Net R50-ViT implementation expects one image channel per slice; "
                f"got {input_channels_per_slice}."
            )
        self.input_channels_per_slice = int(input_channels_per_slice)
        self.num_input_slices = 3

        config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
        config.n_classes = int(num_classes)
        config.n_skip = 3
        config.patches.grid = (image_size // 16, image_size // 16)
        self.model = VisionTransformer(config, img_size=image_size, num_classes=num_classes)

        resolved = Path(pretrained_path) if pretrained_path else None
        if resolved is not None and resolved.is_file():
            with np.load(resolved) as weights:
                self.model.load_from(weights=weights)
            self.pretrained_path = str(resolved)
        elif os.environ.get("BHSD_CSA_ALLOW_RANDOM_INIT", "0") == "1":
            self.pretrained_path = None
        else:
            expected = resolved or Path("R50+ViT-B_16.npz")
            raise FileNotFoundError(
                "CSA-Net requires the ImageNet-21k R50+ViT-B_16 weights used by the official code. "
                f"Missing: {expected}. Set BHSD_CSA_PRETRAINED to the uploaded .npz file."
            )

    @property
    def decoder(self):
        return self.model.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, 3*C, H, W], got {tuple(x.shape)}")
        expected_channels = self.num_input_slices * self.input_channels_per_slice
        if x.shape[1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, got {x.shape[1]}")
        previous, center, following = torch.split(x, self.input_channels_per_slice, dim=1)
        return self.model(previous, center, following)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]):
        if len(input_size) != 2:
            raise AssertionError("CSA-Net expects a 2D spatial patch size")
        return 0

from __future__ import annotations

from nnunet25d.common.dataloader_25d import nnUNetDataLoader25D


class Stage3TripletDataLoader(nnUNetDataLoader25D):
    """Locked real-triplet loader; R0 duplication happens after augmentation in the wrapper."""

    def __init__(self, *args, num_input_slices: int = 3, **kwargs):
        if num_input_slices != 3:
            raise ValueError("Stage3 is preregistered for exactly three consecutive slices")
        super().__init__(*args, num_input_slices=3, **kwargs)


__all__ = ["Stage3TripletDataLoader"]

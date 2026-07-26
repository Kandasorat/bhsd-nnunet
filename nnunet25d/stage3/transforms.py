from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


_SYNC_REQUIRED_CLASS_NAMES = frozenset(
    {
        "GaussianNoiseTransform",
        "GaussianBlurTransform",
        "MultiplicativeBrightnessTransform",
        "ContrastTransform",
        "SimulateLowResolutionTransform",
        "GammaTransform",
    }
)


@dataclass(frozen=True)
class IntensitySynchronizationRecord:
    class_name: str
    synchronize_channels: bool


def iter_transform_tree(transform: Any) -> Iterator[Any]:
    """Yield every transform in a batchgeneratorsv2 transform tree once."""

    seen: set[int] = set()
    stack = [transform]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attribute in ("transforms", "list_of_transforms"):
            children = getattr(current, attribute, None)
            if children is not None:
                stack.extend(reversed(tuple(children)))
        child = getattr(current, "transform", None)
        if child is not None:
            stack.append(child)


def synchronize_stage3_intensity_transforms(transform: Any) -> tuple[IntensitySynchronizationRecord, ...]:
    """Force one sampled intensity parameter set across the three CT slices."""

    records = []
    found = set()
    for item in iter_transform_tree(transform):
        class_name = type(item).__name__
        if class_name not in _SYNC_REQUIRED_CLASS_NAMES:
            continue
        if not hasattr(item, "synchronize_channels"):
            raise RuntimeError(f"{class_name} no longer exposes synchronize_channels")
        item.synchronize_channels = True
        # batchgeneratorsv2 synchronizes sampled values but still samples the
        # per-channel application mask independently. Stage3 requires the
        # outer RandomTransform decision to apply one operation to all slices.
        if hasattr(item, "p_per_channel"):
            item.p_per_channel = 1.0
        found.add(class_name)
        records.append(IntensitySynchronizationRecord(class_name, bool(item.synchronize_channels)))

    missing = _SYNC_REQUIRED_CLASS_NAMES - found
    if missing:
        raise RuntimeError(f"Stage3 transform pipeline is missing required intensity transforms: {sorted(missing)}")
    assert_stage3_intensity_synchronization(transform)
    return tuple(records)


def assert_stage3_intensity_synchronization(transform: Any) -> None:
    inspected = []
    for item in iter_transform_tree(transform):
        class_name = type(item).__name__
        if class_name in _SYNC_REQUIRED_CLASS_NAMES:
            inspected.append(class_name)
            if getattr(item, "synchronize_channels", None) is not True:
                raise RuntimeError(f"Stage3 requires {class_name}.synchronize_channels=True")
            if hasattr(item, "p_per_channel") and float(item.p_per_channel) != 1.0:
                raise RuntimeError(f"Stage3 requires {class_name}.p_per_channel=1")
    missing = _SYNC_REQUIRED_CLASS_NAMES - set(inspected)
    if missing:
        raise RuntimeError(f"Could not audit required intensity transforms: {sorted(missing)}")


def required_synchronized_transform_names() -> tuple[str, ...]:
    return tuple(sorted(_SYNC_REQUIRED_CLASS_NAMES))

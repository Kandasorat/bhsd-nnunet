from __future__ import annotations

import time
from contextlib import nullcontext
from collections.abc import Callable

import numpy as np
import torch
from torch import nn


def conv2d_flops(module: nn.Module, inputs: tuple, output) -> int:
    if not isinstance(module, nn.Conv2d):
        return 0
    output_tensor = output[0] if isinstance(output, (list, tuple)) else output
    batch, output_channels, output_height, output_width = output_tensor.shape
    kernel_height, kernel_width = module.kernel_size
    operations_per_output = 2 * (module.in_channels // module.groups) * kernel_height * kernel_width
    if module.bias is not None:
        operations_per_output += 1
    return int(batch * output_channels * output_height * output_width * operations_per_output)


def measure_conv2d_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    total = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal total
        total += conv2d_flops(module, inputs, output)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(input_tensor)
    finally:
        for handle in handles:
            handle.remove()
    return total


def measure_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    warmup: int = 10,
    repetitions: int = 50,
    use_autocast: bool = False,
) -> dict:
    if warmup < 1 or repetitions < 1:
        raise ValueError("warmup and repetitions must be positive")
    device = input_tensor.device
    model = model.to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    flops = measure_conv2d_flops(model, input_tensor)
    def autocast_context():
        if use_autocast and device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=True)
        return nullcontext()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            with autocast_context():
                model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(repetitions):
            start = time.perf_counter()
            with autocast_context():
                model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return {
        "device": str(device),
        "input_shape": list(input_tensor.shape),
        "parameters": int(parameters),
        "trainable_parameters": int(trainable),
        "conv2d_flops_batch": int(flops),
        "flop_convention": "multiply and add count as two operations; Conv2d only",
        "warmup": warmup,
        "repetitions": repetitions,
        "autocast": bool(use_autocast and device.type == "cuda"),
        "median_latency_ms": float(np.median(timings)),
        "peak_allocated_bytes": peak_memory,
    }


def count_complete_backbone_passes(model: nn.Module, center: nn.Module, input_tensor: torch.Tensor) -> int:
    calls = 0

    def hook(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = center.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_tensor)
    finally:
        handle.remove()
    return calls

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nnunet25d.attention.unified_slice_adapters import METHODS, UnifiedSliceAdapter


TRAINERS = (
    "nnUNetTrainer_25D_AdapterControl",
    "nnUNetTrainer_25D_CSAMSliceGate",
    "nnUNetTrainer_25D_ECASliceGate",
    "nnUNetTrainer_25D_PixelWiseCrossSlice",
    "nnUNetTrainer_25D_CSACenterNeighbor",
    "nnUNetTrainer_25D_CBAM",
    "nnUNetTrainer_25D_CoordinateAttention",
    "nnUNetTrainer_25D_AxialSliceConv",
)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Identity()
        self.head = nn.Conv2d(3, 6, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def compute_conv_feature_map_size(self, _input_size):
        return 0


def run_unit_checks(device: torch.device) -> None:
    for method in sorted(METHODS):
        torch.manual_seed(3407)
        model = UnifiedSliceAdapter(TinyBackbone(), method=method).to(device)
        model.eval()
        x = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)

        adapted = model.adapted_input(x)
        if not torch.equal(adapted, x):
            raise AssertionError(f"{method}: adapter is not exact identity at initialization")

        with torch.no_grad():
            model.project.weight.fill_(0.01)
        first = model(x).detach()[0]
        changed = x.detach().clone()
        changed[1].add_(100.0)
        second = model(changed).detach()[0]
        if not torch.allclose(first, second, atol=1e-6, rtol=1e-6):
            raise AssertionError(f"{method}: one sample affected another sample")

        output = model(x)
        if output.shape != (2, 6, 32, 32):
            raise AssertionError(f"{method}: unexpected output shape {tuple(output.shape)}")
        output.square().mean().backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise AssertionError(f"{method}: invalid input gradient")
        mechanism_gradients = [p.grad for p in model.mechanism.parameters() if p.requires_grad]
        if mechanism_gradients:
            finite = [g for g in mechanism_gradients if g is not None and torch.isfinite(g).all()]
            if not finite:
                raise AssertionError(f"{method}: mechanism received no finite gradient")
            if not any(torch.count_nonzero(g).item() > 0 for g in finite):
                raise AssertionError(f"{method}: mechanism is stuck at zero gradient")

        restored = UnifiedSliceAdapter(TinyBackbone(), method=method).to(device).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            restored_output = restored(x.detach())
        if not torch.allclose(output.detach(), restored_output, atol=1e-6, rtol=1e-6):
            raise AssertionError(f"{method}: strict checkpoint round trip changed the output")

        adapter_parameters = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("backbone."))
        print(f"PASS unit method={method} adapter_parameters={adapter_parameters}")


def first_target(batch: dict) -> torch.Tensor:
    target = batch["target"]
    return target[0] if isinstance(target, list) else target


def run_real_data_checks(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("--real-data is intended for CUDA verification")
    from nnunetv2.run.run_training import get_trainer_from_args

    for trainer_name in TRAINERS:
        torch.cuda.reset_peak_memory_stats(device)
        trainer = get_trainer_from_args("Dataset001_BHSD", "2d", 0, trainer_name, device=device)
        trainer.initialize()
        train_loader, _ = trainer.get_dataloaders()
        batch = next(train_loader)
        trainer.network.train()
        step_result = trainer.train_step(batch)
        loss = float(step_result["loss"])
        if not torch.isfinite(torch.tensor(loss)):
            raise AssertionError(f"{trainer_name}: non-finite train-step loss {loss}")
        data = batch["data"].to(device, non_blocking=False)
        target = first_target(batch)
        trainer.network.eval()
        with torch.no_grad():
            output = trainer.network(data[:1])
            output = output[0] if isinstance(output, (list, tuple)) else output
        if data.shape[1:] != (3, 256, 256):
            raise AssertionError(f"{trainer_name}: unexpected data shape {tuple(data.shape)}")
        if target.shape[-2:] != (256, 256):
            raise AssertionError(f"{trainer_name}: unexpected target shape {tuple(target.shape)}")
        if output.shape != (1, 6, 256, 256):
            raise AssertionError(f"{trainer_name}: unexpected output shape {tuple(output.shape)}")
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(
            f"PASS real trainer={trainer_name} data={tuple(data.shape)} "
            f"target={tuple(target.shape)} output={tuple(output.shape)} loss={loss:.6f} "
            f"peak_cuda_mb={peak_mb:.1f}"
        )
        del output, data, batch, train_loader, trainer
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all harmonized 2.5D attention-screen adapters")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--real-data", action="store_true", help="also initialize every Dataset001 fold-0 trainer")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    run_unit_checks(device)
    if args.real_data:
        run_real_data_checks(device)
    print("ALL_ATTENTION_SCREEN_CHECKS_PASSED")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import zoom
from torch import nn
from torch.utils.data import DataLoader

from nnunet25d.csam.CSAM_networks import C2BAMUNet
from nnunet25d.csam.CSAM_modules import SliceAttentionModule
from nnunet25d.csa_net.official_wrapper import OfficialCSANet3SliceWrapper
from source_faithful.bhsd_data import (
    BHSDCaseStore,
    CSAMSequenceDataset,
    CSANetSliceDataset,
    fold_keys,
    resize_pair,
)


class UpstreamCSAM(nn.Module):
    """Released CSAM U-Net defaults: 20 slices, six levels, base width 64."""

    def __init__(self, num_classes: int, sequence_length: int, base_width: int = 64):
        super().__init__()
        self.model = C2BAMUNet(
            input_channels=1,
            num_classes=num_classes,
            num_layers=6,
            base_num=base_width,
            batch_size=sequence_length,
            semantic=True,
            positional=True,
            slice=True,
            uncertainty=True,
            rank=5,
        )
        for module in self.model.modules():
            if isinstance(module, SliceAttentionModule):
                module.source_faithful_sampling = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


class UpstreamDiceLoss(nn.Module):
    """CSA-Net's released Dice loss, including its all-class/(C-1) convention."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target.long(), self.num_classes).movedim(-1, 1).float()
        loss = logits.new_tensor(0.0)
        for class_index in range(self.num_classes):
            score, truth = scores[:, class_index], one_hot[:, class_index]
            intersect = torch.sum(score * truth)
            denominator = torch.sum(score * score) + torch.sum(truth * truth)
            loss = loss + 1.0 - (2.0 * intersect + 1e-5) / (denominator + 1e-5)
        return loss / (self.num_classes - 1)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def foreground_macro_dice(predictions: list[np.ndarray], targets: list[np.ndarray], classes: int) -> dict:
    per_class = []
    for label in range(1, classes):
        intersection = 0
        denominator = 0
        for prediction, target in zip(predictions, targets):
            pred_mask, target_mask = prediction == label, target == label
            intersection += int(np.logical_and(pred_mask, target_mask).sum())
            denominator += int(pred_mask.sum() + target_mask.sum())
        per_class.append(float(2 * intersection / denominator) if denominator else 1.0)
    return {"per_class_dice": per_class, "foreground_mean_dice": float(np.mean(per_class))}


def validate_csanet(model, store, keys, size, classes, device, max_cases=None) -> dict:
    model.eval()
    predictions, targets = [], []
    selected = keys if max_cases is None else keys[:max_cases]
    with torch.no_grad():
        for key in selected:
            image, target = store.load(key)
            case_prediction = []
            for z in range(image.shape[0]):
                indices = (max(0, z - 1), z, min(image.shape[0] - 1, z + 1))
                tensors = []
                for index in indices:
                    resized, _ = resize_pair(image[index], target[z], size)
                    tensors.append(torch.from_numpy(np.ascontiguousarray(resized[None, None])).float().to(device))
                pred = model(torch.cat(tensors, dim=1)).argmax(1)[0].cpu().numpy()
                if pred.shape != target[z].shape:
                    pred = zoom(pred, (target.shape[1] / pred.shape[0], target.shape[2] / pred.shape[1]), order=0)
                case_prediction.append(pred)
            predictions.append(np.stack(case_prediction))
            targets.append(target)
    return foreground_macro_dice(predictions, targets, classes)


def validate_csam(model, store, keys, length, size, classes, device, max_cases=None) -> dict:
    # The released CSAM samples its uncertainty distribution during validation.
    # train() preserves that behavior; no gradients are recorded below.
    model.train()
    predictions, targets = [], []
    selected = keys if max_cases is None else keys[:max_cases]
    with torch.no_grad():
        for key in selected:
            image, target = store.load(key)
            logits_sum = np.zeros((classes, *target.shape), dtype=np.float32)
            counts = np.zeros(target.shape[0], dtype=np.float32)
            last = max(0, image.shape[0] - length)
            starts = list(range(0, last + 1, max(1, length // 2)))
            if not starts or starts[-1] != last:
                starts.append(last)
            for start in starts:
                indices = np.clip(np.arange(start, start + length), 0, image.shape[0] - 1)
                slices = [resize_pair(image[z], target[z], size)[0] for z in indices]
                tensor = torch.from_numpy(np.ascontiguousarray(np.stack(slices)[:, None])).float().to(device)
                logits = model(tensor).cpu().numpy()
                if logits.shape[-2:] != target.shape[-2:]:
                    logits = zoom(
                        logits,
                        (1, 1, target.shape[1] / logits.shape[2], target.shape[2] / logits.shape[3]),
                        order=1,
                    )
                for local, z in enumerate(indices):
                    logits_sum[:, z] += logits[local]
                    counts[z] += 1
            predictions.append((logits_sum / counts[None, :, None, None]).argmax(0))
            targets.append(target)
    return foreground_macro_dice(predictions, targets, classes)


def save_checkpoint(path: Path, model, optimizer, epoch: int, best: float, config: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_foreground_mean_dice": best,
            "config": config,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    os.replace(temporary, path)


def train(config: dict, smoke: bool, resume: bool) -> None:
    set_seed(int(config["seed"]), bool(config.get("deterministic", False)))
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    preprocessed = Path(os.environ["nnUNet_preprocessed"]) / config["dataset_name"]
    train_keys, val_keys = fold_keys(preprocessed, int(config["fold"]))
    train_store = BHSDCaseStore(preprocessed, train_keys)
    val_store = BHSDCaseStore(preprocessed, val_keys)
    if smoke:
        smoke_root = Path(os.environ.get("BHSD_SOURCE_SMOKE_DIR", Path.cwd() / "results" / "source_faithful_smoke"))
        output = smoke_root / config["experiment_name"]
    else:
        output = Path(config["output_dir"])
        existing = list(output.glob("checkpoint_*.pth")) if output.exists() else []
        if existing and not resume:
            raise FileExistsError(
                f"Refusing to mix or overwrite an existing source-faithful run in {output}: {existing}"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    classes = int(config["num_classes"])

    if config["method"] == "csam":
        length, size = int(config["sequence_length"]), int(config["input_size"])
        reduced_smoke = smoke and os.environ.get("BHSD_EXACT_SMOKE", "0") != "1"
        base_width = int(config.get("smoke_base_width", 4) if reduced_smoke else config.get("base_width", 64))
        model = UpstreamCSAM(classes, length, base_width).to(device)
        dataset = CSAMSequenceDataset(train_store, train_keys[:1] if smoke else train_keys, length, size)
        loader = DataLoader(
            dataset,
            batch_size=1 if smoke else int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
            num_workers=0 if smoke else int(config.get("num_workers", 1)),
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        criterion = nn.CrossEntropyLoss()
    else:
        size = int(config.get("smoke_input_size", 64)) if smoke and device.type == "cpu" else int(config["input_size"])
        pretrained = config["pretrained_path"]
        if smoke and not Path(pretrained).is_file():
            os.environ["BHSD_CSA_ALLOW_RANDOM_INIT"] = "1"
            pretrained = None
        model = OfficialCSANet3SliceWrapper(1, classes, size, pretrained).to(device)
        dataset = CSANetSliceDataset(train_store, train_keys[:1] if smoke else train_keys, size, augment=True)
        loader = DataLoader(
            dataset,
            batch_size=1 if smoke else int(config["batch_size"]),
            shuffle=True,
            num_workers=0 if smoke else int(config.get("num_workers", 8)),
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config["learning_rate"]), momentum=0.9, weight_decay=1e-4)
        dice_loss = UpstreamDiceLoss(classes)

    if smoke:
        batch = next(iter(loader))
        model.train()
        optimizer.zero_grad()
        if config["method"] == "csam":
            losses = [
                criterion(model(sequence.to(device)), label.to(device))
                for sequence, label in zip(batch["image"], batch["label"])
            ]
            loss = torch.stack(losses).mean()
        else:
            logits = model(
                torch.cat(
                    (batch["prev"].to(device), batch["image"].to(device), batch["next"].to(device)),
                    dim=1,
                )
            )
            label = batch["label"].to(device)
            loss = 0.5 * F.cross_entropy(logits, label) + 0.5 * dice_loss(logits, label)
        loss.backward()
        payload = {
            "method": config["method"],
            "device": str(device),
            "effective_input_size": size,
            "effective_base_width": base_width if config["method"] == "csam" else None,
            "loss": float(loss),
            "smoke": "passed",
        }
        (output / "smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return

    epochs, best, history, start_epoch = int(config["epochs"]), float("-inf"), [], 0
    latest = output / "checkpoint_latest.pth"
    if resume:
        if not latest.is_file():
            raise FileNotFoundError(f"--resume requested but checkpoint is missing: {latest}")
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint.get("config", {}).get("experiment_name") != config["experiment_name"]:
            raise RuntimeError("Checkpoint experiment_name does not match the requested source-faithful config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_foreground_mean_dice"])
        history_path = output / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

    max_iterations = epochs * len(loader)
    iteration = start_epoch * len(loader)
    for epoch in range(start_epoch, epochs):
        model.train()
        running = []
        for batch in loader:
            optimizer.zero_grad()
            if config["method"] == "csam":
                losses = [
                    criterion(model(sequence.to(device)), label.to(device))
                    for sequence, label in zip(batch["image"], batch["label"])
                ]
                loss = torch.stack(losses).mean()
            else:
                logits = model(
                    torch.cat(
                        (batch["prev"].to(device), batch["image"].to(device), batch["next"].to(device)),
                        dim=1,
                    )
                )
                label = batch["label"].to(device)
                loss = 0.5 * F.cross_entropy(logits, label) + 0.5 * dice_loss(logits, label)
            loss.backward()
            optimizer.step()
            if config["method"] == "csa_net":
                lr = float(config["learning_rate"]) * (1.0 - iteration / max_iterations) ** 0.9
                for group in optimizer.param_groups:
                    group["lr"] = lr
            iteration += 1
            running.append(float(loss.detach().cpu()))

        should_validate = config["method"] == "csam" or (epoch > 10 and (epoch % 5 == 0 or epoch == epochs - 1))
        row = {"epoch": epoch, "train_loss": float(np.mean(running))}
        if should_validate:
            if config["method"] == "csam":
                metrics = validate_csam(model, val_store, val_keys, int(config["sequence_length"]), size, classes, device)
            else:
                metrics = validate_csanet(model, val_store, val_keys, size, classes, device)
            row.update(metrics)
            if metrics["foreground_mean_dice"] > best:
                best = metrics["foreground_mean_dice"]
                save_checkpoint(output / "checkpoint_best.pth", model, optimizer, epoch, best, config)
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_checkpoint(output / "checkpoint_latest.pth", model, optimizer, epoch, best, config)
        print(json.dumps(row))

    save_checkpoint(output / "checkpoint_final.pth", model, optimizer, epochs - 1, best, config)
    evaluated = [row for row in history if "foreground_mean_dice" in row]
    best_row = max(evaluated, key=lambda row: row["foreground_mean_dice"])
    (output / "summary.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = Path(args.config)
    if not path.is_file():
        path = Path(__file__).resolve().parents[1] / "configs" / f"{args.config}.yaml"
    train(yaml.safe_load(path.read_text(encoding="utf-8")), args.smoke, args.resume)


if __name__ == "__main__":
    main()

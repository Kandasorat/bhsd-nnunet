from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import evaluate, infer, load_config  # noqa: E402


BASE_CONFIGS = {
    "d1": "spectral_25d_d1_lowpass_fold0",
    "d5": "spectral_25d_d5_adaptive_oriented_fold0",
    "d6": "spectral_25d_d6_adaptive_invariant_fold0",
}
PREDICTION_MODES = ("original", "swapped", "swap_average")


def run_method(method: str) -> None:
    base = load_config(BASE_CONFIGS[method])
    for mode in PREDICTION_MODES:
        config = deepcopy(base)
        config["experiment_name"] = f"spectral_direction_probe_{method}_{mode}_fold0_seed3407"
        config["protocol_tier"] = "inference_only_spectral_direction_probe"
        config["spectral_prediction_mode"] = mode
        config["save_probabilities"] = False
        config["continue_prediction"] = False
        config["notes"] = [
            "Inference-only reuse of the existing checkpoint; no retraining.",
            f"Direction probe mode: {mode}.",
            "Original versus swapped measures direction sensitivity; swap_average measures group averaging.",
        ]
        print(f"Running {method} checkpoint in {mode} mode")
        infer(config)
        evaluate(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate trained D1/D5/D6 checkpoints under original, swapped, and averaged directions."
    )
    parser.add_argument("--method", choices=sorted(BASE_CONFIGS), required=True)
    args = parser.parse_args()
    run_method(args.method)


if __name__ == "__main__":
    main()

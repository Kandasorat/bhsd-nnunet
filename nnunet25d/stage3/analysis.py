from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nnunet25d.stage3.statistics import (
    patient_cluster_bootstrap_present_delta,
    rows_from_dicts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked Stage3 patient-cluster confirmatory analysis")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--treatment", default="R1")
    parser.add_argument("--control", default="R0")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    args = parser.parse_args()
    if args.iterations != 10_000 or args.seed != 20_260_726:
        raise ValueError("Confirmatory bootstrap iterations and seed are preregistered and cannot be changed")
    with Path(args.input_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    rows = rows_from_dicts(records)
    result = patient_cluster_bootstrap_present_delta(
        rows,
        args.treatment,
        args.control,
        iterations=args.iterations,
        seed=args.seed,
    )
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

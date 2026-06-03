from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


FOLD_CASE_METRICS_PATTERN = re.compile(r".+_fold_\d+_case_metrics\.csv$")


def collect_case_metrics(results_root: Path, output_csv: Path) -> pd.DataFrame:
    csv_files = sorted(p for p in results_root.glob("*/*_case_metrics.csv") if p.is_file())
    fold_case_csvs = [p for p in csv_files if FOLD_CASE_METRICS_PATTERN.fullmatch(p.name)]
    selected_csvs = fold_case_csvs if fold_case_csvs else csv_files
    frames = []
    for csv_file in selected_csvs:
        if csv_file.name == output_csv.name:
            continue
        frame = pd.read_csv(csv_file)
        if "experiment_name" not in frame.columns:
            frame["experiment_name"] = csv_file.parent.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No case metric CSV files found under {results_root}")
    merged = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    collect_case_metrics(Path(args.results_root), Path(args.output_csv))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compare_models(input_csv: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    summary = (
        df.groupby("model")[metric]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    summary = compare_models(Path(args.input_csv), args.metric)
    summary.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()

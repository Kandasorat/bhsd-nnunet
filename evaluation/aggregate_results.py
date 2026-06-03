from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate_case_metrics(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    identifier_columns = {"case_id", "fold", "model", "class_id", "class_name", "experiment_name"}
    metric_columns = [
        c
        for c in df.columns
        if c not in identifier_columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    group_cols = [c for c in ["model", "class_id", "class_name"] if c in df.columns]
    if not group_cols:
        group_cols = ["model"] if "model" in df.columns else []

    if group_cols:
        agg = df.groupby(group_cols)[metric_columns].agg(["mean", "std"]).reset_index()
        agg.columns = ["_".join([str(i) for i in col if i]).rstrip("_") for col in agg.columns.to_flat_index()]
    else:
        agg = pd.DataFrame({f"{metric}_mean": [df[metric].mean()] for metric in metric_columns})
        for metric in metric_columns:
            agg[f"{metric}_std"] = df[metric].std(ddof=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_csv, index=False)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    aggregate_case_metrics(Path(args.input_csv), Path(args.output_csv))


if __name__ == "__main__":
    main()

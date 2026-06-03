from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def paired_tests(input_csv: Path, metric: str, baseline: str, contender: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    subset = df[df["model"].isin([baseline, contender])]
    index_columns = [c for c in ["fold", "case_id", "class_id", "class_name"] if c in subset.columns]
    if not index_columns:
        index_columns = ["case_id"] if "case_id" in subset.columns else []
    pivot = subset.pivot_table(index=index_columns, columns="model", values=metric, aggfunc="mean")
    pivot = pivot.dropna()

    baseline_values = pivot[baseline]
    contender_values = pivot[contender]

    if len(pivot) == 0:
        raise ValueError(f"No paired samples found for {baseline} vs {contender} using metric '{metric}'.")

    wilcoxon_stat, wilcoxon_p = wilcoxon(baseline_values, contender_values)
    t_stat, t_p = ttest_rel(baseline_values, contender_values)

    return pd.DataFrame(
        [
            {
                "metric": metric,
                "baseline": baseline,
                "contender": contender,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
                "ttest_stat": t_stat,
                "ttest_p": t_p,
                "n_pairs": len(pivot),
                "pairing_keys": ",".join(index_columns),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--contender", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    output = paired_tests(Path(args.input_csv), args.metric, args.baseline, args.contender)
    output.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()

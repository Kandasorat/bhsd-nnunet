from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--metric", default="dice")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    summary = df.groupby(["model", "class_name"])[args.metric].mean().reset_index()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="class_name", y=args.metric, hue="model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=300)
    plt.savefig(output_prefix.with_suffix(".svg"))
    summary.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)


if __name__ == "__main__":
    main()

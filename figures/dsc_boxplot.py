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
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="model", y=args.metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=300)
    plt.savefig(output_prefix.with_suffix(".svg"))


if __name__ == "__main__":
    main()

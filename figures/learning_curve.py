from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--x-col", default="epoch")
    parser.add_argument("--y-cols", nargs="+", default=["train_loss", "val_loss"])
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for column in args.y_cols:
        if column in df.columns:
            plt.plot(df[args.x_col], df[column], label=column)
    plt.xlabel(args.x_col)
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=300)
    plt.savefig(output_prefix.with_suffix(".svg"))


if __name__ == "__main__":
    main()

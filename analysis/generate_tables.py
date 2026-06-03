from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def generate_markdown_table(input_csv: Path, output_md: Path) -> None:
    df = pd.read_csv(input_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(df.to_markdown(index=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    generate_markdown_table(Path(args.input_csv), Path(args.output_md))


if __name__ == "__main__":
    main()

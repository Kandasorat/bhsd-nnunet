from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--baseline-model", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(project_root / "analysis" / "compare_models.py"),
            "--input-csv",
            args.input_csv,
            "--metric",
            args.metric,
            "--output-csv",
            str(output_dir / "model_comparison.csv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root / "analysis" / "generate_tables.py"),
            "--input-csv",
            str(output_dir / "model_comparison.csv"),
            "--output-md",
            str(output_dir / "model_comparison.md"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root / "figures" / "dsc_boxplot.py"),
            "--input-csv",
            args.input_csv,
            "--output-prefix",
            str(output_dir / "dsc_boxplot"),
            "--metric",
            args.metric,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root / "figures" / "class_comparison.py"),
            "--input-csv",
            args.input_csv,
            "--output-prefix",
            str(output_dir / "class_comparison"),
            "--metric",
            args.metric,
        ],
        check=True,
    )

    if args.baseline_model is not None:
        import pandas as pd

        df = pd.read_csv(args.input_csv)
        models = sorted(set(df["model"]) - {args.baseline_model})
        for contender in models:
            subprocess.run(
                [
                    sys.executable,
                    str(project_root / "evaluation" / "statistical_tests.py"),
                    "--input-csv",
                    args.input_csv,
                    "--metric",
                    args.metric,
                    "--baseline",
                    args.baseline_model,
                    "--contender",
                    contender,
                    "--output-csv",
                    str(output_dir / f"stats_{args.baseline_model}_vs_{contender}.csv"),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()

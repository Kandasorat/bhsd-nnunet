from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run_plot(script: Path, input_csv: Path, output_prefix: Path) -> None:
    subprocess.run(
        [sys.executable, str(script), "--input-csv", str(input_csv), "--output-prefix", str(output_prefix)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-script", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    run_plot(Path(args.plot_script), Path(args.input_csv), Path(args.output_prefix))


if __name__ == "__main__":
    main()

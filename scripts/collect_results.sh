#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUTPUT_CSV="${PROJECT_ROOT}/results/aggregated/all_case_metrics.csv"

python "${PROJECT_ROOT}/analysis/collect_results.py" \
  --results-root "${PROJECT_ROOT}/results" \
  --output-csv "${OUTPUT_CSV}"

echo "Collected case metrics into ${OUTPUT_CSV}"

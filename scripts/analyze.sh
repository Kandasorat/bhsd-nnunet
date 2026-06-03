#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
INPUT_CSV="${1:?Usage: bash scripts/analyze.sh <combined_case_metrics_csv> [baseline_model] [metric]}"
BASELINE_MODEL="${2:-}"
METRIC="${3:-dice}"
OUTPUT_DIR="${PROJECT_ROOT}/results/reports"

ARGS=(
  --input-csv "$INPUT_CSV"
  --output-dir "$OUTPUT_DIR"
  --metric "$METRIC"
)

if [[ -n "$BASELINE_MODEL" ]]; then
  ARGS+=(--baseline-model "$BASELINE_MODEL")
fi

python "${PROJECT_ROOT}/analysis/build_report.py" "${ARGS[@]}"

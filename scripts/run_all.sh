#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

CONFIGS=(
  baseline_2d
  baseline_3d
)

for config_name in "${CONFIGS[@]}"; do
  echo "Running ${config_name}"
  python "${PROJECT_ROOT}/scripts/run_experiment.py" run_all --config "${config_name}"
done

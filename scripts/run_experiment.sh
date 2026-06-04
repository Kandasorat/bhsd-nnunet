#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
CONFIG_NAME="${1:?Usage: bash scripts/run_experiment.sh <config_name>}"

if [[ "${CONFIG_NAME}" == naive_25d_* || "${CONFIG_NAME}" == spacing_aware_25d ]]; then
  echo "Custom 2.5D configs currently support training only."
  echo "Use: python scripts/run_experiment.py train --config ${CONFIG_NAME}"
  exit 1
fi

python "${PROJECT_ROOT}/scripts/run_experiment.py" run_all --config "${CONFIG_NAME}"

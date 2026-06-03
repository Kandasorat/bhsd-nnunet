#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
CONFIG_NAME="${1:?Usage: bash scripts/evaluate.sh <config_name>}"

python "${PROJECT_ROOT}/scripts/run_experiment.py" evaluate --config "${CONFIG_NAME}"

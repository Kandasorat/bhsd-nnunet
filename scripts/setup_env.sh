#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

conda env create -f "${PROJECT_ROOT}/environment.yml" || conda env update -f "${PROJECT_ROOT}/environment.yml"

echo "Conda environment is ready."
echo "Activate it with: conda activate bhsd-nnunet"

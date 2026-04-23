#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Running scaling law analysis..."

python ./src/scaling_laws/analysis/fit_scaling_laws.py \
  --config ./configs/analysis.yaml

echo "Scaling law analysis completed."

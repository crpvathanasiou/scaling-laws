#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Running compute frontier analysis..."

python ./src/scaling_laws/analysis/compute_frontier.py \
  --config ./configs/analysis.yaml

echo "Compute frontier analysis completed."

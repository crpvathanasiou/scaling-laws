#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Generating training curve figures..."

python ./src/scaling_laws/analysis/plot_training_curves.py \
  --config ./configs/training_curves.yaml

echo "Training curve figures completed."

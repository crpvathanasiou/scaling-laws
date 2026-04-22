#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Running model training..."

python ./src/scaling_laws/train/train_model.py \
  --config ./configs/train_base.yaml

echo "Model training completed."

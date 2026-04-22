#!/usr/bin/env bash
set -euo pipefail

# Move to repository root, no matter where the script is called from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Running tokenizer profiling + final tokenizer training..."

python ./src/scaling_laws/tokenizer/train_tokenizer.py \
  --config ./configs/tokenizer.yaml

echo "Tokenizer run completed."
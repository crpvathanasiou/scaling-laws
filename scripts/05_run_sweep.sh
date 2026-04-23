#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-small}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "$MODE" in
  small)
    CONFIG_PATH="./configs/sweep_small.yaml"
    DRY_RUN=false
    ;;
  medium)
    CONFIG_PATH="./configs/sweep_medium.yaml"
    DRY_RUN=false
    ;;
  large)
    CONFIG_PATH="./configs/sweep_large.yaml"
    DRY_RUN=false
    ;;
  all)
    CONFIG_PATH="./configs/sweep.yaml"
    DRY_RUN=false
    ;;
  dry-run)
    CONFIG_PATH="./configs/sweep.yaml"
    DRY_RUN=true
    ;;
  *)
    echo "Invalid mode: $MODE"
    echo "Use one of: small | medium | large | all | dry-run"
    exit 1
    ;;
esac

echo "Running sweep mode: $MODE"

if [ "$DRY_RUN" = true ]; then
  python ./src/scaling_laws/experiments/run_sweep.py \
    --config "$CONFIG_PATH" \
    --dry_run
else
  python ./src/scaling_laws/experiments/run_sweep.py \
    --config "$CONFIG_PATH"
fi

echo "Sweep completed."

#!/usr/bin/env bash
set -euo pipefail

# Train the Nakagami Fig.7.2-style CNN+DNN on the exported 4M dataset.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_PARENT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

GPU="${GPU:-0}"
DATA_DIR="${DATA_DIR:-dataset/nakagami_atrest_voxel_gnn_4M}"
DATASET_TAG="${DATASET_TAG:-nakagami_fig72_cnndnn_4M}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-200}"
LR="${LR:-4e-5}"
DROPOUT="${DROPOUT:-0.3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FILE="${LOG_FILE:-$HOME/train_nakagami_fig72_cnndnn_4M.log}"
RESUME_ARG=()
LIMIT_ARGS=()

if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARG=(--resume "$RESUME")
fi
if [[ -n "${MAX_TRAIN_EVENTS:-}" ]]; then
  LIMIT_ARGS+=(--max-train-events "$MAX_TRAIN_EVENTS")
fi
if [[ -n "${MAX_VAL_EVENTS:-}" ]]; then
  LIMIT_ARGS+=(--max-val-events "$MAX_VAL_EVENTS")
fi
if [[ -n "${MAX_TEST_EVENTS:-}" ]]; then
  LIMIT_ARGS+=(--max-test-events "$MAX_TEST_EVENTS")
fi
if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  LIMIT_ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "${MAX_VAL_BATCHES:-}" ]]; then
  LIMIT_ARGS+=(--max-val-batches "$MAX_VAL_BATCHES")
fi

CUDA_VISIBLE_DEVICES="$GPU" python -u src/train/train_nakagami_cnndnn.py \
  --data-dir "$DATA_DIR" \
  --dataset-tag "$DATASET_TAG" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --dropout "$DROPOUT" \
  --num-workers "$NUM_WORKERS" \
  --amp \
  "${RESUME_ARG[@]}" \
  "${LIMIT_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"

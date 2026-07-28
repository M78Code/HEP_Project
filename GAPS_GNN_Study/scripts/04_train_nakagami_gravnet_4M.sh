#!/usr/bin/env bash
set -euo pipefail

# Train GravNet on sparse nodes extracted from Nakagami fixed-grid voxels.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_PARENT:${PYTHONPATH:-}"

GPU="${GPU:-0}"
DATA_DIR="${DATA_DIR:-dataset/nakagami_atrest_voxel_gnn_4M}"
DATASET_TAG="${DATASET_TAG:-nakagami_atrest_sparse_voxel_gravnet_4M_tof11}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FILE="${LOG_FILE:-$HOME/train_nakagami_atrest_sparse_voxel_gravnet_4M_tof11.log}"

CUDA_VISIBLE_DEVICES="$GPU" python -u src/train/train_nakagami_gravnet.py \
  --data-dir "$DATA_DIR" \
  --dataset-tag "$DATASET_TAG" \
  --model gravnet \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --hidden 128 \
  --num-blocks 6 \
  --k 8 \
  --dropout 0.15 \
  --amp \
  --min-epochs 20 \
  --early-stopping-patience 20 \
  --early-stopping-min-delta 1e-5 \
  2>&1 | tee "$LOG_FILE"

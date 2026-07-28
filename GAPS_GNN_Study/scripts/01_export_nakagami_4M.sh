#!/usr/bin/env bash
set -euo pipefail

# Export Nakagami 1457-column CSV files to the shared npy format used by
# CNN+DNN and sparse-voxel GNN training.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_PARENT:${PYTHONPATH:-}"

INPUTS="${INPUTS:-/mnt/ynakagami2/SimulationData/210713_renew_topIso_flat/csvFiles}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset/nakagami_atrest_voxel_gnn_4M}"
EVENTS_PER_CLASS="${EVENTS_PER_CLASS:-2000000}"
LOG_FILE="${LOG_FILE:-$HOME/export_nakagami_atrest_voxel_gnn_4M.log}"

python -u src/data/export_nakagami_csv_to_voxel.py \
  --inputs "$INPUTS" \
  --events-per-class "$EVENTS_PER_CLASS" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

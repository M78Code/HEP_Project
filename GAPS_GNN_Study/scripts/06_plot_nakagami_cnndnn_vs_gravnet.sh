#!/usr/bin/env bash
set -euo pipefail

# Plot the Rejection Curve comparison for Nakagami CNN+DNN and GravNet.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_PARENT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

CNNDNN_EVAL="${CNNDNN_EVAL:-results/20260720-183528_CNNDNNFig72_nakagami_fig72_cnndnn_4M_rerun/evaluation_test}"
DATASET_TAG="${DATASET_TAG:-nakagami_atrest_sparse_voxel_gravnet_4M_tof11}"
if [[ -z "${GRAVNET_EVAL:-}" ]]; then
  LATEST_DIR="$(find results -maxdepth 1 -type d -name "*_${DATASET_TAG}" | sort | tail -n 1)"
  if [[ -z "$LATEST_DIR" ]]; then
    echo "No GravNet evaluation found for DATASET_TAG=${DATASET_TAG}. Run scripts/04 and scripts/05 first or set GRAVNET_EVAL." >&2
    exit 1
  fi
  GRAVNET_EVAL="${LATEST_DIR}/evaluation_test"
fi
OUT_DIR="${OUT_DIR:-results/nakagami_cnndnn_vs_gravnet_4M_tof11_compare}"
X_MIN="${X_MIN:-0.5}"
Y_MAX="${Y_MAX:-1e6}"

python src/plot/plot_rejection_compare.py \
  --item "CNN+DNN" "$CNNDNN_EVAL" \
  --item "GravNet" "$GRAVNET_EVAL" \
  --out-dir "$OUT_DIR" \
  --x-min "$X_MIN" \
  --y-max "$Y_MAX"

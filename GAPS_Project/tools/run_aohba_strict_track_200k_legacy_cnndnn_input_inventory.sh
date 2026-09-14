#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
STRICT_CACHE=${STRICT_CACHE:-/mnt/aohba/archive_20260913_after_submission_mc1/aohba_stopping_ablation_strict_track_200k_global_log}
HISTORICAL_CHECKPOINT=${HISTORICAL_CHECKPOINT:-"$PROJECT/nakagami/results/nakagami4M_cnndnn_best.pth"}
LEGACY_DATASET=${LEGACY_DATASET:-/mnt/aohba/aohba4M_voxel3input_from_pkl}
OUT_DIR=${OUT_DIR:-"$PROJECT/results/aohba_strict_track_200k_legacy_cnndnn_input_inventory"}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
cd "$PROJECT"

python -u src/scripts/audit_strict_track_legacy_cnndnn_inputs.py \
    --strict-cache "$STRICT_CACHE" \
    --historical-checkpoint "$HISTORICAL_CHECKPOINT" \
    --legacy-dataset "$LEGACY_DATASET" \
    --out-dir "$OUT_DIR"

echo "report: $OUT_DIR/legacy_cnndnn_input_inventory.md"

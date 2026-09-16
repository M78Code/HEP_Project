#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
PROVENANCE=${PROVENANCE:-/mnt/aohba/aohba_treerec_hit_proxy_calibration_candidates_300k}
OUTPUT=${OUTPUT:-"$PROJECT/results/aohba_treerec_proxy_disagreement_audit_300k/audit.json"}
SEED=${SEED:-20260825}

cd "$PROJECT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka

echo "===== TreeRec proxy disagreement audit ====="
echo "provenance : $PROVENANCE"
echo "output     : $OUTPUT"
echo "GPU        : not used"

python -u src/data_parse/audit_treerec_proxy_disagreement.py \
    --provenance-dir "$PROVENANCE" \
    --output "$OUTPUT" \
    --seed "$SEED"

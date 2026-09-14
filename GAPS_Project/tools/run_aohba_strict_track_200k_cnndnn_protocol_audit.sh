#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
DATASET=${DATASET:-/mnt/aohba/aohba_2tof_strict_track_stop_top_trigger_200k_treerec_cnndnn_10x12x12}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_strict_track_200k_cnndnn"}
OUT_DIR=${OUT_DIR:-"$RESULT_ROOT/protocol_audit"}
REFERENCE_CHECKPOINT=${REFERENCE_CHECKPOINT:-"$PROJECT/nakagami/results/nakagami4M_cnndnn_best.pth"}
TAG=${TAG:-aohba_strict_track_stop_treerec_200k_cnndnn_seed20260825}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
cd "$PROJECT"

latest_run() {
    find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name "*_CNNDNNFig72_${TAG}" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-
}

test -f "$DATASET/_SUCCESS" || {
    echo "ERROR: strict TreeRec CNN+DNN dataset is missing: $DATASET" >&2
    exit 1
}

CURRENT_RUN=$(latest_run)
test -n "$CURRENT_RUN" || {
    echo "ERROR: completed strict 200K CNN+DNN run not found in $RESULT_ROOT" >&2
    exit 1
}
test -f "$CURRENT_RUN/best.pt" || {
    echo "ERROR: best checkpoint not found: $CURRENT_RUN/best.pt" >&2
    exit 1
}

REFERENCE_ARGS=()
if test -f "$REFERENCE_CHECKPOINT"; then
    REFERENCE_ARGS=(--reference-checkpoint "$REFERENCE_CHECKPOINT")
else
    echo "WARNING: historical CNN+DNN checkpoint not found; running dataset/current-run audit only." >&2
fi

echo "current strict CNN+DNN run: $CURRENT_RUN"
echo "current strict dataset    : $DATASET"
echo "historical reference      : ${REFERENCE_ARGS[*]:-not supplied}"

python -u src/scripts/audit_strict_track_cnndnn_protocol.py \
    --dataset "$DATASET" \
    --split-suffix cnndnn_10x12x12 \
    --current-run "$CURRENT_RUN" \
    --out-dir "$OUT_DIR" \
    "${REFERENCE_ARGS[@]}"

echo "report: $OUT_DIR/cnndnn_protocol_audit.md"

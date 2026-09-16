#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "ERROR: bootstrap runner stopped at line $LINENO (exit=$status)" >&2' ERR

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_treerec_proxy_fourway_20k"}
OUT_DIR=${OUT_DIR:-"$RESULT_ROOT/comparison/bootstrap_high_efficiency_ci"}
REPEATS=${REPEATS:-5000}
SEED=${SEED:-20260825}
CACHE_PREFIX=${CACHE_PREFIX:-aohba_treerec_proxy_fourway}
EVENTS_PER_CLASS=${EVENTS_PER_CLASS:-20000}
GROUPS=(
    summary_atrest_topology
    treerec_logistic_proxy
    treerec_full_graph_oof_proxy
    truth_strict_topology
)

cd "$PROJECT"
echo "[START] four-way high-efficiency bootstrap CI"
echo "project : $PROJECT"
echo "results : $RESULT_ROOT"
echo "repeats : $REPEATS"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka

latest_run_dir() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -name "*_${2}" -print 2>/dev/null | sort | tail -1
}

args=()
for group in "${GROUPS[@]}"; do
    tag="${CACHE_PREFIX}_${group}_${EVENTS_PER_CLASS}_global_log_seed${SEED}"
    group_root="$RESULT_ROOT/$group"
    [[ -d "$group_root" ]] || { echo "ERROR: result directory missing: $group_root" >&2; exit 1; }
    echo "[RESOLVE] $group"
    run_dir=$(latest_run_dir "$group_root" "$tag" || true)
    eval_dir="$run_dir/evaluation_test"
    [[ -n "$run_dir" && -f "$eval_dir/labels.npy" && -f "$eval_dir/scores.npy" ]] || {
        echo "ERROR: evaluation missing for $group" >&2; exit 1;
    }
    case "$group" in
        summary_atrest_topology) label="Summary at-rest + Umbrella-to-Cube" ;;
        treerec_logistic_proxy) label="TreeRec hitseries logistic proxy" ;;
        treerec_full_graph_oof_proxy) label="TreeRec full-graph OOF proxy" ;;
        truth_strict_topology) label="Truth strict stop (K=0 + zero step)" ;;
    esac
    args+=(--item "$label" "$eval_dir")
done

python -u src/scripts/visual/bootstrap_binary_eval_ci.py "${args[@]}" \
    --out-dir "$OUT_DIR" --repeats "$REPEATS" --seed "$SEED" \
    --targets 0.90 0.95 0.98 0.99

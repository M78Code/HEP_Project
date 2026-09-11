#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
OUT=${OUT:-"$PROJECT/results/aohba_selection_ablation_200k_comparison"}

NO_SELECTION="$PROJECT/results/aohba_pair200k_digitization_ab/20260910-132942_GravNet_6b_h128_aohba_pair200k_provided_global_log_seed20260825/evaluation_test"
TOP_TRIGGER="$PROJECT/results/aohba_toptrigger_only_treerec_200k/20260911-160330_GravNet_6b_h128_aohba_toptrigger_only_treerec_200k_global_log_seed20260825/evaluation_test"
AT_REST_TOP_TRIGGER="$PROJECT/results/aohba_fixedgrid_selected_treerec_200k/20260911-141452_GravNet_6b_h128_aohba_fixedgrid_selected_treerec_200k_global_log_seed20260825/evaluation_test"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
cd "$PROJECT"

for directory in "$NO_SELECTION" "$TOP_TRIGGER" "$AT_REST_TOP_TRIGGER"; do
    [[ -f "$directory/labels.npy" && -f "$directory/scores.npy" ]] || {
        echo "ERROR: evaluation arrays missing: $directory" >&2
        exit 1
    }
done

python src/scripts/visual/compare_binary_eval.py \
    --item "No secondary selection" "$NO_SELECTION" \
    --item "Top-trigger only" "$TOP_TRIGGER" \
    --item "At-rest + top-trigger" "$AT_REST_TOP_TRIGGER" \
    --out-dir "$OUT" \
    --x-min 0.50 \
    --y-max 20000 \
    --mark-efficiencies 0.90 0.95 0.98 0.99

echo "figure : $OUT/rejection_compare.png"
echo "metrics: $OUT/metrics_compare.json"
echo "AOHBA SELECTION ABLATION 200K COMPARISON: COMPLETE"

#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}

export PROJECT
export CANDIDATES=${CANDIDATES:-/mnt/aohba/aohba_stopping_summary_only_candidates_300k}
export MATCHED=${MATCHED:-/mnt/aohba/aohba_stopping_summary_only_beta_matched_200k}
export LOGDIR=${LOGDIR:-"$HOME/aohba_stopping_summary_only_200k_logs"}
export RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_stopping_summary_only_ablation_200k"}
export CANDIDATES_PER_CLASS=${CANDIDATES_PER_CLASS:-300000}
export EVENTS_PER_CLASS=${EVENTS_PER_CLASS:-100000}
export GPU=${GPU:-0}
export SEED=${SEED:-20260825}

export GROUP_A=strict_track
export GROUP_B=summary_only
export SELECTION_A=stopped-toptrigger
export SELECTION_B=summary-only-toptrigger
export LABEL_A="Strict track stop + top-trigger"
export LABEL_B="Stopping-volume summary only + top-trigger"

exec bash "$PROJECT/tools/run_aohba_stopping_ablation_200k.sh" "$@"

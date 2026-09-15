#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
export PROJECT
export CANDIDATES=/mnt/aohba/aohba_legacy_atrest_vs_strict_candidates_300k
export MATCHED=/mnt/aohba/aohba_legacy_atrest_vs_strict_beta_matched_200k
export LOGDIR="$HOME/aohba_legacy_atrest_vs_strict_200k_logs"
export RESULT_ROOT="$PROJECT/results/aohba_legacy_atrest_vs_strict_200k"
export CACHE_PREFIX=aohba_legacy_atrest_vs_strict
export CANDIDATES_PER_CLASS=300000
export EVENTS_PER_CLASS=100000
export GPU=0
export SEED=20260825

export GROUP_A=strict_track_topology
export GROUP_B=legacy_atrest
export SELECTION_A=stopped-toptrigger
export SELECTION_B=legacy-atrest
export LABEL_A="Strict track stop + Umbrella-to-Cube"
export LABEL_B="Legacy tracker-at-rest summary"

exec bash "$PROJECT/tools/run_aohba_stopping_ablation_200k.sh" "$@"

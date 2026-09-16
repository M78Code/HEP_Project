#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
MATCHED=${MATCHED:-/mnt/aohba/aohba_strict_stop_evidence_beta_matched_100k_per_class}
RESULT=${RESULT:-"$PROJECT/results/aohba_treerec_stop_proxy_audit_20k"}
EVENTS_PER_CLASS=${EVENTS_PER_CLASS:-20000}

cd "$PROJECT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka

for group in summary_atrest_topology strict_track_topology; do
    for particle in antiP antiD; do
        test -f "$MATCHED/$group/$particle/_SUCCESS" || {
            echo "ERROR: missing matched provenance: $MATCHED/$group/$particle" >&2
            exit 1
        }
    done
done

mkdir -p "$RESULT"
python -u src/data_parse/audit_treerec_stop_proxy.py \
    --events-per-class "$EVENTS_PER_CLASS" \
    --group "summary_atrest_topology=$MATCHED/summary_atrest_topology" \
    --group "strict_track_topology=$MATCHED/strict_track_topology" \
    --output "$RESULT/treerec_stop_proxy_audit.json"

echo "result: $RESULT/treerec_stop_proxy_audit.json"

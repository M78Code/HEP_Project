#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
cd "$PROJECT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
source "$HOME/setup_ynakagami_root.sh"

mkdir -p build/tools

g++ -std=c++17 -O2 $(root-config --cflags) \
    -I/home/ynakagami/simpledet/SimpleDet/common/include \
    -I/home/ynakagami/simpledet/build/install/gaps-v1.7.0/include/gaps \
    tools/export/export_treemc_topiso_like_csv.cc \
    -L/home/ynakagami/simpledet/build/common -lGAPSCommon \
    $(root-config --libs) \
    -o build/tools/export_treemc_topiso_like_csv

EXE="$PROJECT/build/tools/export_treemc_topiso_like_csv"
OUT=$(mktemp -d /tmp/m78code_aohba_treemc_direct_equivalence.XXXXXX)
printf '%s\n' "$OUT" | tee "$HOME/aohba_treemc_direct_equivalence_latest.path"

PRELOAD="/home/ynakagami/GEANT/install/lib/libG4geometry.so:/home/ynakagami/GEANT/install/lib/libG4event.so:/home/ynakagami/simpledet/build/common/libGAPSCommon.so"

run_particle() {
    local particle=$1
    local label=$2
    local input=$3
    local csv="$OUT/${particle}.csv"
    local npy="$OUT/${particle}_direct"

    echo "===== $particle: legacy CSV ====="
    env LD_PRELOAD="$PRELOAD" "$EXE" \
        --input "$input" \
        --geometry-file "$input" \
        --output "$csv" \
        --max-events 1000 \
        --target-label "$label" \
        >"$OUT/${particle}_csv.log" 2>&1

    echo "===== $particle: direct NPY ====="
    env LD_PRELOAD="$PRELOAD" "$EXE" \
        --input "$input" \
        --geometry-file "$input" \
        --output-npy-dir "$npy" \
        --max-events 1000 \
        --target-label "$label" \
        >"$OUT/${particle}_direct.log" 2>&1

    echo "===== $particle: equivalence audit ====="
    python src/scripts/audit_treemc_fixedgrid_direct_export.py \
        --csv "$csv" \
        --npy-dir "$npy"
}

run_particle \
    antiP \
    0 \
    /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root

run_particle \
    antiD \
    1 \
    /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root

echo
du -sh "$OUT"
echo "output: $OUT"
echo "AOHBA TREEMC DIRECT NPY EQUIVALENCE: PASSED"

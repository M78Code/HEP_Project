#!/usr/bin/env bash
set -euo pipefail
cd ~/HEP_Project/GAPS_Project/nakagami/..
python nakagami/data_parse/preprocess_onlyprimary_4M.py \
  --base-dir /mnt/ynakagami2/SimulationData/210922_trigger/csvFiles_onlyPrimary/atrest_shuffled \
  --out-dir /mnt/ynakagami3/nakagami_data/data_4M_onlyprimary

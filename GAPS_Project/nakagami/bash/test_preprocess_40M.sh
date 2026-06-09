#!/bin/bash
# 中上40M CSV 前処理 - 小サンプルテスト
# 目的: pipeline動作確認（各クラス2ファイル）
# 実行: bash nakagami/bash/test_preprocess_40M.sh

OUT_DIR=~/HEP_Project/GAPS_Project/nakagami/data_40M_test
SCRIPT=~/HEP_Project/GAPS_Project/nakagami/data_parse/preprocess_40M.py
LOG=~/HEP_Project/GAPS_Project/nakagami/bash/test_preprocess_40M.log

DBAR_DIR=/mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles
PBAR_DIR=/mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Mc

mkdir -p $OUT_DIR

echo "Start: $(date)" | tee $LOG
echo "Dbar dir: $DBAR_DIR" | tee -a $LOG
echo "Pbar dir: $PBAR_DIR" | tee -a $LOG
echo "Out dir : $OUT_DIR"  | tee -a $LOG
echo "max_files_per_class: 2 (テスト)" | tee -a $LOG

python $SCRIPT \
    --csv_dirs $DBAR_DIR $PBAR_DIR \
    --out_dir  $OUT_DIR \
    --max_files_per_class 2 \
    --train_ratio 0.8 \
    --num_workers 12 \
    2>&1 | tee -a $LOG

# npz内容を検証
python -c "
import numpy as np
import os
for f in ['train_hybrid_nakagami40M.npz', 'val_hybrid_nakagami40M.npz']:
    path = os.path.expanduser('~/HEP_Project/GAPS_Project/nakagami/data_40M_test/') + f
    if not os.path.exists(path):
        print(f'{f}: not found')
        continue
    d = np.load(path)
    print(f'{f}:')
    print(f'  keys      : {list(d.keys())}')
    print(f'  voxels    : {d[\"voxels\"].shape}  dtype={d[\"voxels\"].dtype}')
    print(f'  tofs      : {d[\"tofs\"].shape}    dtype={d[\"tofs\"].dtype}')
    print(f'  labels    : {d[\"labels\"].shape}  dtype={d[\"labels\"].dtype}')
    u, c = np.unique(d['labels'], return_counts=True)
    print(f'  label dist: {dict(zip(u.tolist(), c.tolist()))}')
    print(f'  tof[0]    : {d[\"tofs\"][0]}')
" 2>&1 | tee -a $LOG

echo "End: $(date)" | tee -a $LOG

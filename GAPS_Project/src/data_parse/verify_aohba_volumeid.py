"""
大場 2tof データの volume_id 分布を確認し,Si(Li) と TOF (Umbrella/Cortina/Cube) を切り分ける。
- TreeRec/hitseries_.volume_id_ の unique 値とその個数を出力
- 既存 voxelizer の volume_id 仕様(GAPS_Project/src/data_parse/voxelizer.py)と照合する材料
- 1 ファイルから 5000 events サンプルすれば十分
"""

import uproot
import numpy as np
from pathlib import Path
from collections import Counter

ROOT_DIR = Path('/mnt/aohba/GAPS_Sim_2tof')
N_EVENTS = 5000     # 各粒子 1 ファイルから抽出

def inspect_volume_ids(particle):
    files = sorted((ROOT_DIR / particle).glob('*.root'))
    fp = files[0]
    print(f'\n========== {particle}: {fp.name} ==========')

    with uproot.open(fp) as f:
        # 最新 cycle の TreeRec を取る
        rec_key = sorted(k for k in f.keys() if k.startswith('TreeRec;'))[-1]
        rec = f[rec_key]

        # hitseries の必要 branch だけ読む(高速化)
        vol = rec['Rec/hitseries_/hitseries_.volume_id_'].array(entry_stop=N_EVENTS, library='np')
        edep = rec['Rec/hitseries_/hitseries_.energydep_'].array(entry_stop=N_EVENTS, library='np')

    # event 単位 jagged → 全 hit を平坦化
    all_vol = np.concatenate([v for v in vol if len(v) > 0])
    all_edep = np.concatenate([e for e in edep if len(e) > 0])
    print(f'total hits in {N_EVENTS} events: {len(all_vol):,}')
    print(f'avg hits/event: {len(all_vol) / N_EVENTS:.2f}')

    # volume_id の頻度 top 30
    cnt = Counter(all_vol.tolist())
    print(f'\nunique volume_ids: {len(cnt)}')
    print(f'  most common (top 30):')
    for vid, c in cnt.most_common(30):
        # その volume_id の平均 dE
        mask = all_vol == vid
        mean_e = all_edep[mask].mean()
        print(f'    vol_id={vid:>12d}  hits={c:>8d}  mean_dE={mean_e:.3e}')

    # volume_id の range / グループ化目安
    arr = np.array(sorted(cnt.keys()))
    print(f'\nvolume_id range: [{arr.min():d}, {arr.max():d}]')
    # 桁ごとにビン化(1e7, 1e8, 1e9...で TOF/Si(Li) が分かれる可能性)
    for thresh in [1e5, 1e6, 1e7, 1e8, 1e9]:
        n_below = (arr < thresh).sum()
        if 0 < n_below < len(arr):
            print(f'  unique vol_ids < {thresh:.0e}: {n_below} / {len(arr)}')

if __name__ == '__main__':
    for p in ['antiD', 'antiP']:
        inspect_volume_ids(p)
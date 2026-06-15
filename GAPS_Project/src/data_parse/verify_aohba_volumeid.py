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


"""
(naka) m78code@gp1:~/HEP_Project/GAPS_Project$ python src/data_parse/verify_aohba_volumeid.py 2>&1 | tee ~/aohba_volumeid.log

========== antiD: antiD_2tof_FTFP_BERT_1781253355.root ==========
total hits in 5000 events: 129,167
avg hits/event: 25.83

unique volume_ids: 11361
  most common (top 30):
    vol_id=   110003000  hits=    1303  mean_dE=9.242e+00
    vol_id=   110052000  hits=    1275  mean_dE=9.249e+00
    vol_id=   110053000  hits=    1271  mean_dE=8.805e+00
    vol_id=   110004000  hits=    1255  mean_dE=9.113e+00
    vol_id=   110002000  hits=    1253  mean_dE=9.527e+00
    vol_id=   110054000  hits=    1206  mean_dE=9.216e+00
    vol_id=   100003000  hits=    1175  mean_dE=9.544e+00
    vol_id=   110051000  hits=    1172  mean_dE=9.244e+00
    vol_id=   100052000  hits=    1155  mean_dE=8.604e+00
    vol_id=   110001000  hits=    1128  mean_dE=1.049e+01
    vol_id=   110005000  hits=    1074  mean_dE=1.044e+01
    vol_id=   100053000  hits=    1062  mean_dE=8.919e+00
    vol_id=   100002000  hits=    1030  mean_dE=9.068e+00
    vol_id=   110050000  hits=     997  mean_dE=1.029e+01
    vol_id=   100004000  hits=     962  mean_dE=8.600e+00
    vol_id=   100051000  hits=     929  mean_dE=8.928e+00
    vol_id=   115000000  hits=     928  mean_dE=7.689e+00
    vol_id=   112000000  hits=     919  mean_dE=8.408e+00
    vol_id=   115001000  hits=     883  mean_dE=6.593e+00
    vol_id=   114001000  hits=     880  mean_dE=7.566e+00
    vol_id=   113001000  hits=     871  mean_dE=5.954e+00
    vol_id=   113000000  hits=     871  mean_dE=8.631e+00
    vol_id=   114050000  hits=     862  mean_dE=5.867e+00
    vol_id=   112050000  hits=     858  mean_dE=7.168e+00
    vol_id=   114000000  hits=     857  mean_dE=7.793e+00
    vol_id=   115050000  hits=     856  mean_dE=6.744e+00
    vol_id=   112001000  hits=     839  mean_dE=7.418e+00
    vol_id=   113050000  hits=     829  mean_dE=6.577e+00
    vol_id=   112051000  hits=     824  mean_dE=4.823e+00
    vol_id=   110055000  hits=     807  mean_dE=1.023e+01

volume_id range: [100000000, 209350307]

========== antiP: antiP_2tof_FTFP_BERT_1781424263.root ==========
total hits in 5000 events: 92,551
avg hits/event: 18.51

unique volume_ids: 10570
  most common (top 30):
    vol_id=   110003000  hits=    1099  mean_dE=7.913e+00
    vol_id=   110002000  hits=    1091  mean_dE=8.355e+00
    vol_id=   110051000  hits=    1084  mean_dE=7.668e+00
    vol_id=   110004000  hits=    1068  mean_dE=8.120e+00
    vol_id=   100003000  hits=    1061  mean_dE=8.053e+00
    vol_id=   110052000  hits=    1044  mean_dE=8.371e+00
    vol_id=   110053000  hits=    1026  mean_dE=8.773e+00
    vol_id=   100052000  hits=    1004  mean_dE=7.562e+00
    vol_id=   110001000  hits=     981  mean_dE=8.365e+00
    vol_id=   110054000  hits=     968  mean_dE=9.046e+00
    vol_id=   100002000  hits=     938  mean_dE=7.465e+00
    vol_id=   100053000  hits=     925  mean_dE=7.665e+00
    vol_id=   110050000  hits=     845  mean_dE=8.436e+00
    vol_id=   110005000  hits=     828  mean_dE=8.763e+00
    vol_id=   100004000  hits=     826  mean_dE=7.507e+00
    vol_id=   100051000  hits=     821  mean_dE=7.795e+00
    vol_id=   113001000  hits=     705  mean_dE=6.893e+00
    vol_id=   113000000  hits=     685  mean_dE=7.633e+00
    vol_id=   115050000  hits=     681  mean_dE=6.483e+00
    vol_id=   112050000  hits=     681  mean_dE=5.768e+00
    vol_id=   114000000  hits=     679  mean_dE=8.918e+00
    vol_id=   113050000  hits=     678  mean_dE=6.913e+00
    vol_id=   114050000  hits=     677  mean_dE=7.499e+00
    vol_id=   100001000  hits=     668  mean_dE=7.011e+00
    vol_id=   112000000  hits=     659  mean_dE=7.502e+00
    vol_id=   112001000  hits=     648  mean_dE=5.941e+00
    vol_id=   115001000  hits=     646  mean_dE=6.446e+00
    vol_id=   114001000  hits=     628  mean_dE=6.789e+00
    vol_id=   115000000  hits=     627  mean_dE=8.279e+00
    vol_id=   113051000  hits=     624  mean_dE=5.584e+00

volume_id range: [100000000, 209350306]
"""
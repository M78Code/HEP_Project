"""
大場 2tof データを chunked で前処理する。Step 2: まず 1 file 全 ev で動作確認、
時間 / ディスクを計測して 240 files (~5010 万 events) の総コストを見積もる。

設計方針:
  * uproot.TTree.array(entry_start, entry_stop) で chunk 単位で読み込む
    (5010 万 events × 26 hits/ev を一気にメモリへ載せない)
  * Step 1 で確認した dict 形式 (volume_id, energy, positions, times, label, beta) を踏襲
  * 既存 convert_root_to_pickle (root_file_reader.py) と互換の pickle 形式
  * N <= 1 event は preprocess 段階で skip (knn_graph が不安定)
  * 出力先は 2080 home 配下: ~/aohba_preprocess/{particle}/{root_stem}.pkl
    (/mnt/aohba は読み専用)

使い方:
  # antiD の 1 file 目を全 ev 前処理
  python src/data_parse/preprocess_aohba_chunked.py --particle antiD --file-idx 0

  # antiP の最初 5 ファイル
  python src/data_parse/preprocess_aohba_chunked.py --particle antiP --file-idx 0 --n-files 5

  # chunk size 調整
  python src/data_parse/preprocess_aohba_chunked.py --particle antiD --chunk-size 20000
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import uproot

ROOT_DIR    = Path('/mnt/aohba/GAPS_Sim_2tof')
DEFAULT_OUT = Path.home() / 'aohba_preprocess'
DEFAULT_CHUNK = 10_000

PDG_ANTIPROTON   = -2212
PDG_ANTIDEUTERON = -1000010020


def chunk_to_events(pdg, beta, vol, edep, hpos, htime):
    """1 chunk 分の awkward arrays から event dict のリストを作る。
    各 hit-level branch の長さが一致しない壊れた event は skip して報告する。
    """
    events = []
    n_inconsistent = 0
    for i in range(len(pdg)):
        v = np.array(vol[i], dtype=np.int64)
        e = np.array(edep[i], dtype=np.float32)

        p_raw = hpos[i]
        if len(p_raw) > 0:
            pos = np.stack([
                np.array(p_raw['fX'], dtype=np.float32),
                np.array(p_raw['fY'], dtype=np.float32),
                np.array(p_raw['fZ'], dtype=np.float32),
            ], axis=1)
        else:
            pos = np.zeros((0, 3), dtype=np.float32)

        t = np.array(htime[i], dtype=np.float32)

        # hit 配列の長さチェック(壊れた event を弾く)
        if not (len(v) == len(e) == len(pos) == len(t)):
            n_inconsistent += 1
            continue

        events.append({
            'energy':    e,
            'positions': pos,
            'times':     t,
            'volume_id': v,
            'label':     int(pdg[i]),
            'beta':      float(beta[i]),
            'n_hits':    len(e),
        })
    return events, n_inconsistent


def process_one_file(root_path, output_dir, chunk_size):
    """1 ROOT ファイルを chunked で処理し pkl + summary を出力する。"""
    t0 = time.time()
    out_pkl     = output_dir / f'{root_path.stem}.pkl'
    out_summary = output_dir / f'{root_path.stem}_summary.json'

    all_events = []
    skipped_small = 0
    skipped_inconsistent = 0

    with uproot.open(root_path) as f:
        # uproot の dict access は latest cycle を返すので、これで OK
        mc  = f['TreeMc']
        rec = f['TreeRec']

        n_total = mc.num_entries
        if rec.num_entries != n_total:
            raise RuntimeError(
                f'TreeMc({n_total}) / TreeRec({rec.num_entries}) entries mismatch in {root_path.name}')

        print(f'  [{root_path.name}] total events = {n_total:,}, chunk = {chunk_size}')

        for start in range(0, n_total, chunk_size):
            stop = min(start + chunk_size, n_total)
            t_chunk = time.time()

            pdg   = mc['Mc/primaryPdg_'].array(entry_start=start, entry_stop=stop)
            beta  = mc['Mc/CEventBase/primaryBetaGenerated_'].array(entry_start=start, entry_stop=stop)
            vol   = rec['Rec/hitseries_/hitseries_.volume_id_'].array(entry_start=start, entry_stop=stop)
            edep  = rec['Rec/hitseries_/hitseries_.energydep_'].array(entry_start=start, entry_stop=stop)
            hpos  = rec['Rec/hitseries_/hitseries_.hit_position_'].array(entry_start=start, entry_stop=stop)
            htime = rec['Rec/hitseries_/hitseries_.hit_time_'].array(entry_start=start, entry_stop=stop)

            chunk_events, n_inc = chunk_to_events(pdg, beta, vol, edep, hpos, htime)
            skipped_inconsistent += n_inc
            # N<=1 event は GraphBuilder で knn_graph が壊れるので skip
            kept = [e for e in chunk_events if e['n_hits'] > 1]
            skipped_small += (len(chunk_events) - len(kept))
            all_events.extend(kept)

            dt = time.time() - t_chunk
            print(f'    chunk [{start:>8}, {stop:>8})  '
                  f'kept {len(kept):>6}/{len(chunk_events):>6}  '
                  f'inc={n_inc}  '
                  f'({dt:.1f}s, {len(chunk_events) / max(dt, 1e-9):.0f} ev/s)')

    t_read = time.time() - t0
    print(f'  read+convert done: {len(all_events):,} events ({t_read:.1f}s)')

    # ── pickle 出力 ──
    t_dump = time.time()
    with open(out_pkl, 'wb') as fp:
        pickle.dump({'events': all_events, 'source_file': root_path.name}, fp,
                    protocol=pickle.HIGHEST_PROTOCOL)
    t_dump = time.time() - t_dump
    pkl_mb = out_pkl.stat().st_size / 1024 / 1024
    print(f'  pickle saved: {out_pkl.name}  ({pkl_mb:.1f} MB, {t_dump:.1f}s)')

    # ── summary.json ──
    n_hits = np.array([e['n_hits'] for e in all_events])
    betas  = np.array([e['beta']   for e in all_events])
    labels = np.array([e['label']  for e in all_events])
    uniq, cnt = np.unique(labels, return_counts=True)

    summary = {
        'source_file':       root_path.name,
        'total_events':      int(len(all_events)),
        'skipped_n_le_1':    int(skipped_small),
        'skipped_inconsistent_lengths': int(skipped_inconsistent),
        'label_counts':      dict(zip([str(u) for u in uniq.tolist()], cnt.tolist())),
        'hits_per_event':    {
            'min':  int(n_hits.min()),
            'max':  int(n_hits.max()),
            'mean': float(round(n_hits.mean(), 2)),
            'median': int(np.median(n_hits)),
        },
        'beta': {
            'min':  float(round(betas.min(), 4)),
            'max':  float(round(betas.max(), 4)),
            'mean': float(round(betas.mean(), 4)),
        },
        'pickle_size_mb':    round(pkl_mb, 2),
        'time_read_convert_sec': round(t_read, 1),
        'time_pickle_dump_sec':  round(t_dump, 1),
        'chunk_size':        chunk_size,
    }
    with open(out_summary, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(f'  summary saved: {out_summary.name}')

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--particle', required=True, choices=['antiD', 'antiP'])
    ap.add_argument('--file-idx',  type=int, default=0,
                    help='開始ファイルインデックス (sorted glob 順)')
    ap.add_argument('--n-files',   type=int, default=1,
                    help='処理するファイル数')
    ap.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    files = sorted((ROOT_DIR / args.particle).glob('*.root'))
    targets = files[args.file_idx : args.file_idx + args.n_files]
    if not targets:
        raise SystemExit(f'no files in range idx={args.file_idx}..{args.file_idx + args.n_files - 1}')

    out_dir = args.output_dir / args.particle
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output dir : {out_dir}')
    print(f'particle   : {args.particle}')
    print(f'targets    : {[t.name for t in targets]}')
    print(f'chunk_size : {args.chunk_size}')
    # home 配下に大量出力しないよう注意喚起
    try:
        if str(out_dir).startswith(str(Path.home())) and args.n_files >= 10:
            print(f'\n⚠️  output is under HOME ({Path.home()}). 10+ files で 60GB 制限に近づく可能性。')
            print(f'   --output-dir で /mnt/ynakagami3/... 等を指定することを推奨。')
    except Exception:
        pass
    print()

    t_all = time.time()
    summaries = []
    for fp in targets:
        print(f'\n=== {fp.name} ===')
        summaries.append(process_one_file(fp, out_dir, args.chunk_size))
    t_all = time.time() - t_all

    # ── 全体まとめ + 240 files への外挿 ──
    total_ev = sum(s['total_events'] for s in summaries)
    total_mb = sum(s['pickle_size_mb'] for s in summaries)
    print('\n========== overall ==========')
    print(f'files     : {len(summaries)}')
    print(f'events    : {total_ev:,}')
    print(f'pickle MB : {total_mb:.1f}')
    print(f'wall time : {t_all:.1f}s ({total_ev / max(t_all, 1e-9):.0f} ev/s)')
    print()
    if summaries:
        avg_t = t_all / len(summaries)
        avg_mb = total_mb / len(summaries)
        print('--- extrapolation to full 240 files (~5010万 ev) ---')
        print(f'  estimated wall time : {avg_t * 240 / 60:.1f} min ({avg_t * 240 / 3600:.2f} h)')
        print(f'  estimated disk      : {avg_mb * 240 / 1024:.1f} GB')


if __name__ == '__main__':
    main()


"""
(naka) m78code@gp1:~/HEP_Project/GAPS_Project$ python src/data_parse/preprocess_aohba_chunked.py \
>     --particle antiD --file-idx 0 \
>     2>&1 | tee ~/aohba_step2.log


output dir : /home/m78code/aohba_preprocess/antiD
particle   : antiD
targets    : ['antiD_2tof_FTFP_BERT_1781253355.root']
chunk_size : 10000


=== antiD_2tof_FTFP_BERT_1781253355.root ===
  [antiD_2tof_FTFP_BERT_1781253355.root] total events = 247,721, chunk = 10000
    chunk [       0,    10000)  kept  10000/ 10000  inc=0  (7.5s, 1331 ev/s)
    chunk [   10000,    20000)  kept  10000/ 10000  inc=0  (7.5s, 1334 ev/s)
    chunk [   20000,    30000)  kept  10000/ 10000  inc=0  (8.1s, 1241 ev/s)
    chunk [   30000,    40000)  kept  10000/ 10000  inc=0  (7.7s, 1304 ev/s)
    chunk [   40000,    50000)  kept  10000/ 10000  inc=0  (7.2s, 1382 ev/s)
    chunk [   50000,    60000)  kept  10000/ 10000  inc=0  (6.7s, 1495 ev/s)
    chunk [   60000,    70000)  kept  10000/ 10000  inc=0  (6.9s, 1453 ev/s)
    chunk [   70000,    80000)  kept  10000/ 10000  inc=0  (6.7s, 1487 ev/s)
    chunk [   80000,    90000)  kept  10000/ 10000  inc=0  (6.9s, 1456 ev/s)
    chunk [   90000,   100000)  kept  10000/ 10000  inc=0  (6.6s, 1506 ev/s)
    chunk [  100000,   110000)  kept  10000/ 10000  inc=0  (7.4s, 1360 ev/s)
    chunk [  110000,   120000)  kept  10000/ 10000  inc=0  (7.3s, 1377 ev/s)
    chunk [  120000,   130000)  kept  10000/ 10000  inc=0  (6.6s, 1506 ev/s)
    chunk [  130000,   140000)  kept  10000/ 10000  inc=0  (6.9s, 1447 ev/s)
    chunk [  140000,   150000)  kept  10000/ 10000  inc=0  (6.7s, 1496 ev/s)
    chunk [  150000,   160000)  kept  10000/ 10000  inc=0  (6.8s, 1475 ev/s)
    chunk [  160000,   170000)  kept  10000/ 10000  inc=0  (6.7s, 1487 ev/s)
    chunk [  170000,   180000)  kept  10000/ 10000  inc=0  (7.0s, 1438 ev/s)
    chunk [  180000,   190000)  kept  10000/ 10000  inc=0  (6.7s, 1482 ev/s)
    chunk [  190000,   200000)  kept  10000/ 10000  inc=0  (7.2s, 1395 ev/s)
    chunk [  200000,   210000)  kept  10000/ 10000  inc=0  (6.9s, 1440 ev/s)
    chunk [  210000,   220000)  kept  10000/ 10000  inc=0  (6.7s, 1484 ev/s)
    chunk [  220000,   230000)  kept  10000/ 10000  inc=0  (7.2s, 1392 ev/s)
    chunk [  230000,   240000)  kept  10000/ 10000  inc=0  (6.8s, 1475 ev/s)
    chunk [  240000,   247721)  kept   7721/  7721  inc=0  (5.2s, 1484 ev/s)
  read+convert done: 247,721 events (173.9s)
  pickle saved: antiD_2tof_FTFP_BERT_1781253355.pkl  (204.5 MB, 4.6s)
  summary saved: antiD_2tof_FTFP_BERT_1781253355_summary.json

========== overall ==========
files     : 1
events    : 247,721
pickle MB : 204.5
wall time : 178.8s (1385 ev/s)

--- extrapolation to full 240 files (~5010万 ev) ---
  estimated wall time : 715.2 min (11.92 h)
  estimated disk      : 47.9 GB
  
  

(naka) m78code@gp1:~/HEP_Project/GAPS_Project$ python src/data_parse/preprocess_aohba_chunked.py \
>     --particle antiD --file-idx 0 \
>     2>&1 | tee ~/aohba_step2.log


output dir : /home/m78code/aohba_preprocess/antiD
particle   : antiD
targets    : ['antiD_2tof_FTFP_BERT_1781253355.root']
chunk_size : 10000


=== antiD_2tof_FTFP_BERT_1781253355.root ===
  [antiD_2tof_FTFP_BERT_1781253355.root] total events = 247,721, chunk = 10000
    chunk [       0,    10000)  kept  10000/ 10000  inc=0  (7.5s, 1331 ev/s)
    chunk [   10000,    20000)  kept  10000/ 10000  inc=0  (7.5s, 1334 ev/s)
    chunk [   20000,    30000)  kept  10000/ 10000  inc=0  (8.1s, 1241 ev/s)
    chunk [   30000,    40000)  kept  10000/ 10000  inc=0  (7.7s, 1304 ev/s)
    chunk [   40000,    50000)  kept  10000/ 10000  inc=0  (7.2s, 1382 ev/s)
    chunk [   50000,    60000)  kept  10000/ 10000  inc=0  (6.7s, 1495 ev/s)
    chunk [   60000,    70000)  kept  10000/ 10000  inc=0  (6.9s, 1453 ev/s)
    chunk [   70000,    80000)  kept  10000/ 10000  inc=0  (6.7s, 1487 ev/s)
    chunk [   80000,    90000)  kept  10000/ 10000  inc=0  (6.9s, 1456 ev/s)
    chunk [   90000,   100000)  kept  10000/ 10000  inc=0  (6.6s, 1506 ev/s)
    chunk [  100000,   110000)  kept  10000/ 10000  inc=0  (7.4s, 1360 ev/s)
    chunk [  110000,   120000)  kept  10000/ 10000  inc=0  (7.3s, 1377 ev/s)
    chunk [  120000,   130000)  kept  10000/ 10000  inc=0  (6.6s, 1506 ev/s)
    chunk [  130000,   140000)  kept  10000/ 10000  inc=0  (6.9s, 1447 ev/s)
    chunk [  140000,   150000)  kept  10000/ 10000  inc=0  (6.7s, 1496 ev/s)
    chunk [  150000,   160000)  kept  10000/ 10000  inc=0  (6.8s, 1475 ev/s)
    chunk [  160000,   170000)  kept  10000/ 10000  inc=0  (6.7s, 1487 ev/s)
    chunk [  170000,   180000)  kept  10000/ 10000  inc=0  (7.0s, 1438 ev/s)
    chunk [  180000,   190000)  kept  10000/ 10000  inc=0  (6.7s, 1482 ev/s)
    chunk [  190000,   200000)  kept  10000/ 10000  inc=0  (7.2s, 1395 ev/s)
    chunk [  200000,   210000)  kept  10000/ 10000  inc=0  (6.9s, 1440 ev/s)
    chunk [  210000,   220000)  kept  10000/ 10000  inc=0  (6.7s, 1484 ev/s)
    chunk [  220000,   230000)  kept  10000/ 10000  inc=0  (7.2s, 1392 ev/s)
    chunk [  230000,   240000)  kept  10000/ 10000  inc=0  (6.8s, 1475 ev/s)
    chunk [  240000,   247721)  kept   7721/  7721  inc=0  (5.2s, 1484 ev/s)
  read+convert done: 247,721 events (173.9s)
  pickle saved: antiD_2tof_FTFP_BERT_1781253355.pkl  (204.5 MB, 4.6s)
  summary saved: antiD_2tof_FTFP_BERT_1781253355_summary.json

========== overall ==========
files     : 1
events    : 247,721
pickle MB : 204.5
wall time : 178.8s (1385 ev/s)

--- extrapolation to full 240 files (~5010万 ev) ---
  estimated wall time : 715.2 min (11.92 h)
  estimated disk      : 47.9 GB
(naka) m78code@gp1:~/HEP_Project/GAPS_Project$
(naka) m78code@gp1:~/HEP_Project/GAPS_Project$
(naka) m78code@gp1:~/HEP_Project/GAPS_Project$ df -h /mnt/ynakagami3/ /mnt/aohba/ ~/
Filesystem                       Size  Used Avail Use% Mounted on
192.168.9.22:/volume1/ynakagami  104T   56T   49T  54% /mnt/ynakagami3
192.168.9.22:/volume1/aohba      104T   56T   49T  54% /mnt/aohba
/dev/sda2                        915G  449G  420G  52% /
(naka) m78code@gp1:~/HEP_Project/GAPS_Project$ python src/data_parse/preprocess_aohba_chunked.py \
>   --particle antiP --file-idx 0 \
>   2>&1 | tee ~/aohba_step2_antip.log
output dir : /home/m78code/aohba_preprocess/antiP
particle   : antiP
targets    : ['antiP_2tof_FTFP_BERT_1781424263.root']
chunk_size : 10000


=== antiP_2tof_FTFP_BERT_1781424263.root ===
  [antiP_2tof_FTFP_BERT_1781424263.root] total events = 180,270, chunk = 10000
    chunk [       0,    10000)  kept  10000/ 10000  inc=0  (15.9s, 630 ev/s)
    chunk [   10000,    20000)  kept  10000/ 10000  inc=0  (7.3s, 1373 ev/s)
    chunk [   20000,    30000)  kept  10000/ 10000  inc=0  (8.1s, 1228 ev/s)
    chunk [   30000,    40000)  kept  10000/ 10000  inc=0  (7.3s, 1374 ev/s)
    chunk [   40000,    50000)  kept  10000/ 10000  inc=0  (7.3s, 1371 ev/s)
    chunk [   50000,    60000)  kept  10000/ 10000  inc=0  (6.7s, 1484 ev/s)
    chunk [   60000,    70000)  kept  10000/ 10000  inc=0  (6.8s, 1481 ev/s)
    chunk [   70000,    80000)  kept  10000/ 10000  inc=0  (6.7s, 1492 ev/s)
    chunk [   80000,    90000)  kept  10000/ 10000  inc=0  (6.7s, 1487 ev/s)
    chunk [   90000,   100000)  kept  10000/ 10000  inc=0  (6.9s, 1444 ev/s)
    chunk [  100000,   110000)  kept  10000/ 10000  inc=0  (6.8s, 1478 ev/s)
    chunk [  110000,   120000)  kept  10000/ 10000  inc=0  (6.8s, 1475 ev/s)
    chunk [  120000,   130000)  kept  10000/ 10000  inc=0  (6.7s, 1493 ev/s)
    chunk [  130000,   140000)  kept  10000/ 10000  inc=0  (7.0s, 1431 ev/s)
    chunk [  140000,   150000)  kept  10000/ 10000  inc=0  (6.8s, 1471 ev/s)
    chunk [  150000,   160000)  kept  10000/ 10000  inc=0  (6.7s, 1488 ev/s)
    chunk [  160000,   170000)  kept  10000/ 10000  inc=0  (6.7s, 1494 ev/s)
    chunk [  170000,   180000)  kept  10000/ 10000  inc=0  (6.7s, 1498 ev/s)
    chunk [  180000,   180270)  kept    270/   270  inc=0  (0.3s, 1018 ev/s)
  read+convert done: 180,270 events (134.2s)
  pickle saved: antiP_2tof_FTFP_BERT_1781424263.pkl  (112.7 MB, 3.4s)
  summary saved: antiP_2tof_FTFP_BERT_1781424263_summary.json

========== overall ==========
files     : 1
events    : 180,270
pickle MB : 112.7
wall time : 137.9s (1308 ev/s)

--- extrapolation to full 240 files (~5010万 ev) ---
  estimated wall time : 551.4 min (9.19 h)
  estimated disk      : 26.4 GB
  
  

非常漂亮的结果!所有数字都合理。

  关键数据

  ┌────────────────────────┬──────────────────┬──────────────────────────────────┐
  │          项目          │        值        │               评价               │
  ├────────────────────────┼──────────────────┼──────────────────────────────────┤
  │ events                 │ 247,721          │ 与 verify 一致 ✅                │
  ├────────────────────────┼──────────────────┼──────────────────────────────────┤
  │ skipped (N≤1)          │ 0                │ 数据全保留 ✅                    │
  ├────────────────────────┼──────────────────┼──────────────────────────────────┤
  │ skipped (inconsistent) │ 0                │ hit 长度全一致 ✅                │
  ├────────────────────────┼──────────────────┼──────────────────────────────────┤
  │ pkl size               │ 204.5 MB         │ 压缩比 17×(821GB 原始 → 47.9 GB) │
  ├────────────────────────┼──────────────────┼──────────────────────────────────┤
  │ 速度                   │ 1,385 ev/s       │ 稳定(6.7~8.1 s/10K chunk)        │
  ├────────────────────────┼──────────────────┼──────────────────────────────────┤
  │ 单文件                 │ 178.8 s ≈ 3 分钟 │                                  │
  ├────────────────────────┼──────────────────┼──────────────────────────────────┤
  │ 240 files 外推         │ 11.9 h, 47.9 GB  │                                  │
  └────────────────────────┴──────────────────┴──────────────────────────────────┘
  
  
antiD: 100 files × 204.5 MB ≈ 20.0 GB
antiP: 140 files × 112.7 MB ≈ 15.4 GB
合计: 约 35.4 GB

antiD: 100 × 178.8 s ≈ 5.0 h
antiP: 140 × 137.9 s ≈ 5.4 h
合计: 约 10.3 h

总 events: 约 5000万
总 pkl:    约 35-40 GB
总耗时:    约 10-12 小时
"""
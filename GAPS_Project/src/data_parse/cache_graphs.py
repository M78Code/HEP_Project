"""
pklファイルをPyG Dataオブジェクトに変換し、.ptファイルとしてNASにキャッシュする。
キャッシュ後はGraphBuilderの実行が不要になり、訓練速度が大幅に向上する。

使い方:
  # 全ファイル処理（約5-6時間）
  python src/data_parse/cache_graphs.py \
      --antiD-dir /mnt/ynakagami3/aohba_preprocess/antiD \
      --antiP-dir /mnt/ynakagami3/aohba_preprocess/antiP \
      --output-dir /mnt/ynakagami3/aohba_preprocess/graph_cache

  # smoke test（各1ファイルのみ）
  python src/data_parse/cache_graphs.py \
      --antiD-dir /mnt/ynakagami3/aohba_preprocess/antiD \
      --antiP-dir /mnt/ynakagami3/aohba_preprocess/antiP \
      --output-dir /mnt/ynakagami3/aohba_preprocess/graph_cache \
      --max-files 1
"""
import argparse
import pickle
import time
from pathlib import Path

import torch

import GAPS_Project
from GAPS_Project.src.data_parse.graph_builder import GraphBuilder

PROJECT_ROOT = Path(GAPS_Project.__file__).parent


def cache_one_file(pkl_path: Path, out_dir: Path, builder: GraphBuilder) -> dict:
    out_path = out_dir / (pkl_path.stem + '.pt')
    if out_path.exists():
        print(f'  skip (already cached): {pkl_path.name}')
        return {'skipped': True}

    t0 = time.time()
    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)
    events = payload['events']

    data_list = []
    skipped = 0
    for event in events:
        if event['n_hits'] <= 8:  # knn_graph(k=8) が不安定になる小グラフを除外
            skipped += 1
            continue
        data = builder.build_from_dict(event)
        data_list.append(data)

    torch.save(data_list, out_path)

    # サマリーJSON（approx_len用）
    summary_path = out_path.with_suffix('.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump({'n_graphs': len(data_list), 'source': pkl_path.name}, f)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'  {pkl_path.name} → {len(data_list)} graphs, {size_mb:.1f} MB, {elapsed:.1f}s '
          f'({len(data_list)/max(elapsed,1e-9):.0f} ev/s), skipped={skipped}')
    return {'n_graphs': len(data_list), 'size_mb': size_mb, 'elapsed': elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--antiD-dir',   type=Path, required=True)
    ap.add_argument('--antiP-dir',   type=Path, required=True)
    ap.add_argument('--output-dir',  type=Path, required=True)
    ap.add_argument('--max-files',   type=int, default=None, help='各粒子種の処理ファイル数上限')
    ap.add_argument('--k',           type=int, default=8)
    args = ap.parse_args()

    out_antiD = args.output_dir / 'antiD'
    out_antiP = args.output_dir / 'antiP'
    out_antiD.mkdir(parents=True, exist_ok=True)
    out_antiP.mkdir(parents=True, exist_ok=True)

    builder = GraphBuilder(k=args.k, normalize=True)

    antid_files = sorted(f for f in args.antiD_dir.glob('*.pkl')
                         if not f.name.endswith('_summary.pkl'))
    antip_files = sorted(f for f in args.antiP_dir.glob('*.pkl')
                         if not f.name.endswith('_summary.pkl'))

    if args.max_files is not None:
        antid_files = antid_files[:args.max_files]
        antip_files = antip_files[:args.max_files]

    all_files = [(f, out_antiD) for f in antid_files] + \
                [(f, out_antiP) for f in antip_files]

    print(f'処理対象: antiD={len(antid_files)} files, antiP={len(antip_files)} files')
    print(f'出力先: {args.output_dir}')
    print()

    t_all = time.time()
    total_graphs = 0
    total_mb = 0.0

    for i, (pkl_path, out_dir) in enumerate(all_files, 1):
        particle = 'antiD' if out_dir == out_antiD else 'antiP'
        print(f'[{i}/{len(all_files)}] {particle}/{pkl_path.name}')
        result = cache_one_file(pkl_path, out_dir, builder)
        if not result.get('skipped'):
            total_graphs += result['n_graphs']
            total_mb     += result['size_mb']

    elapsed_all = time.time() - t_all
    print(f'\n========== 完了 ==========')
    print(f'総グラフ数  : {total_graphs:,}')
    print(f'総容量      : {total_mb/1024:.2f} GB')
    print(f'総処理時間  : {elapsed_all/3600:.2f} h')
    print(f'平均速度    : {total_graphs/elapsed_all:.0f} ev/s')


if __name__ == '__main__':
    main()

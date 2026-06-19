"""Split large PyG list caches into smaller files for streaming training.

Example:
  python src/data_parse/shard_graph_cache.py \
      --input-dir dataset/local430_atrest_graph_cache \
      --output-dir dataset/local430_atrest_graph_cache_sharded \
      --shard-size 50000
"""
import argparse
import json
import time
from pathlib import Path

import torch


def shard_split(input_path: Path, output_dir: Path, shard_size: int) -> dict:
    print(f'\n=== {input_path.name} ===', flush=True)
    t0 = time.time()
    data_list = torch.load(input_path, map_location='cpu', weights_only=False)
    load_sec = time.time() - t0
    print(f'loaded {len(data_list):,} graphs in {load_sec:.1f}s', flush=True)

    split = input_path.stem
    n_shards = (len(data_list) + shard_size - 1) // shard_size
    output_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    for shard_idx, start in enumerate(range(0, len(data_list), shard_size)):
        shard = data_list[start:start + shard_size]
        output_path = output_dir / f'{split}_{shard_idx:03d}.pt'
        torch.save(shard, output_path)
        size_bytes = output_path.stat().st_size
        total_bytes += size_bytes

        summary = {
            'source': str(input_path),
            'output': str(output_path),
            'split': split,
            'shard_index': shard_idx,
            'n_shards': n_shards,
            'n_graphs': len(shard),
            'size_mb': round(size_bytes / 1024 / 1024, 2),
        }
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(
            f'[{shard_idx + 1:02d}/{n_shards:02d}] '
            f'{output_path.name}: {len(shard):,} graphs, '
            f'{size_bytes / 1024 / 1024:.1f} MB',
            flush=True,
        )

    elapsed = time.time() - t0
    result = {
        'source': str(input_path),
        'split': split,
        'n_graphs': len(data_list),
        'n_shards': n_shards,
        'shard_size': shard_size,
        'total_size_mb': round(total_bytes / 1024 / 1024, 2),
        'load_sec': round(load_sec, 1),
        'elapsed_sec': round(elapsed, 1),
    }
    del data_list
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--shard-size', type=int, default=50000)
    args = parser.parse_args()

    summaries = []
    for split in args.splits:
        input_path = args.input_dir / f'{split}.pt'
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        summaries.append(
            shard_split(input_path, args.output_dir, args.shard_size))

    with open(args.output_dir / 'summary_all.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print('\n========== complete ==========')
    print(f'output dir  : {args.output_dir}')
    print(f'total graphs: {sum(s["n_graphs"] for s in summaries):,}')
    print(f'total shards: {sum(s["n_shards"] for s in summaries):,}')


if __name__ == '__main__':
    main()

"""Build sharded at-rest graph caches with a fixed 172-D TOF input.

The paddle mapping is derived only from train.pkl and then reused for
validation and test.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from tqdm import tqdm

from GAPS_Project.src.data_parse.graph_builder import GraphBuilder
from GAPS_Project.src.data_parse.tof_paddles import (
    N_TOF_PADDLES,
    collect_tof_paddle_ids,
    make_paddle_index,
    save_paddle_ids,
)


def is_atrest_in_tracker(event: dict) -> bool:
    stopping_vol = int(event.get('stopping_vol', 0))
    return (stopping_vol // 1_000_000) >= 200


def load_events(path: Path):
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and 'events' in payload:
        return payload['events']
    return payload


def save_shard(data_list, split: str, shard_idx: int, output_dir: Path) -> dict:
    output_path = output_dir / f'{split}_{shard_idx:03d}.pt'
    torch.save(data_list, output_path)
    summary = {
        'split': split,
        'shard_index': shard_idx,
        'n_graphs': len(data_list),
        'size_mb': round(output_path.stat().st_size / 1024 / 1024, 2),
        'features': {
            'node_dim': 8,
            'graph_feat_dim': 45,
            'tof_paddle_dim': N_TOF_PADDLES,
            'tof_layer_116_fixed': True,
        },
    }
    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(
        f'  saved {output_path.name}: {len(data_list):,} graphs, '
        f'{summary["size_mb"]:.1f} MB',
        flush=True,
    )
    return summary


def cache_split(events, split: str, output_dir: Path, builder: GraphBuilder,
                shard_size: int, min_hits: int, max_events: int | None) -> dict:
    t0 = time.time()
    total = min(len(events), max_events) if max_events is not None else len(events)
    shards = []
    data_list = []
    label_counts = {0: 0, 1: 0}
    skipped_not_atrest = 0
    skipped_small = 0
    skipped_error = 0

    for event in tqdm(events[:total], desc=f'cache {split}', dynamic_ncols=True):
        if not is_atrest_in_tracker(event):
            skipped_not_atrest += 1
            continue
        if int(event.get('n_hits', 0)) <= min_hits:
            skipped_small += 1
            continue
        try:
            data = builder.build_from_dict(event)
        except KeyError:
            raise
        except Exception as exc:
            skipped_error += 1
            if skipped_error <= 5:
                print(
                    f'\n  graph build error #{skipped_error}: '
                    f'{type(exc).__name__}: {exc}',
                    flush=True,
                )
            continue
        label = int(data.y.item())
        label_counts[label] += 1
        data_list.append(data)
        if len(data_list) >= shard_size:
            shards.append(save_shard(
                data_list, split, len(shards), output_dir))
            data_list = []

    if data_list:
        shards.append(save_shard(data_list, split, len(shards), output_dir))

    result = {
        'split': split,
        'source_events': total,
        'n_graphs': sum(row['n_graphs'] for row in shards),
        'n_shards': len(shards),
        'label_counts': {str(k): v for k, v in label_counts.items()},
        'skipped_not_atrest': skipped_not_atrest,
        'skipped_n_le_min_hits': skipped_small,
        'skipped_error': skipped_error,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--shard-size', type=int, default=50_000)
    parser.add_argument('--min-hits', type=int, default=8)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--max-events', type=int, default=None,
                        help='smoke test limit per split')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.split_dir / 'train.pkl'
    print(f'loading paddle mapping source: {train_path}', flush=True)
    train_events = load_events(train_path)
    paddle_ids = collect_tof_paddle_ids(train_events)
    if len(paddle_ids) != N_TOF_PADDLES:
        raise RuntimeError(
            f'expected {N_TOF_PADDLES} train TOF paddles, got {len(paddle_ids)}')
    mapping_path = args.output_dir / 'paddle_ids.json'
    save_paddle_ids(paddle_ids, mapping_path)
    print(f'paddle mapping saved: {mapping_path}', flush=True)

    builder = GraphBuilder(
        k=args.k,
        normalize=True,
        tof_paddle_index=make_paddle_index(paddle_ids),
    )
    summaries = []
    for split in args.splits:
        path = args.split_dir / f'{split}.pkl'
        events = train_events if split == 'train' else load_events(path)
        print(f'\n=== {split}: {path} ===', flush=True)
        summaries.append(cache_split(
            events, split, args.output_dir, builder,
            args.shard_size, args.min_hits, args.max_events,
        ))
        if split == 'train':
            train_events = None

    with open(args.output_dir / 'summary_all.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)
    print(f'\ncomplete: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()

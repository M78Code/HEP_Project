"""Build resumable sharded at-rest GravNet caches for the Aohba dataset.

Each source PKL is processed independently. The aligned MC sidecar is used
only to select tracker-at-rest events and is not stored as a model feature.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from GAPS_Project.src.data_parse.graph_builder import GraphBuilder
from GAPS_Project.src.data_parse.tof_paddles import (
    N_TOF_PADDLES,
    make_paddle_index,
    save_paddle_ids,
    tof_paddle_id,
)


def load_events(path: Path):
    with open(path, 'rb') as file:
        payload = pickle.load(file)
    if isinstance(payload, dict) and 'events' in payload:
        return payload['events']
    return payload


def particle_from_path(path: Path) -> str:
    if path.parent.name in {'antiD', 'antiP'}:
        return path.parent.name
    if 'antiD' in path.stem:
        return 'antiD'
    if 'antiP' in path.stem:
        return 'antiP'
    raise ValueError(f'cannot infer particle from path: {path}')


def sidecar_path(metadata_dir: Path, pkl_path: Path) -> Path:
    return metadata_dir / particle_from_path(pkl_path) / f'{pkl_path.stem}.npz'


def collect_train_paddle_ids(train_paths: list[Path]) -> list[int]:
    paddle_ids = set()
    for index, path in enumerate(train_paths, 1):
        events = load_events(path)
        for event in events:
            volume_ids = np.asarray(event['volume_id'], dtype=np.int64)
            tof_volume_ids = volume_ids[
                (volume_ids // 100_000_000) == 1
            ]
            paddle_ids.update(
                tof_paddle_id(volume_id) for volume_id in tof_volume_ids)
        print(
            f'  paddle scan [{index:03d}/{len(train_paths):03d}] '
            f'{path.name}: unique={len(paddle_ids)}',
            flush=True,
        )
        if len(paddle_ids) > N_TOF_PADDLES:
            raise RuntimeError(
                f'found more than {N_TOF_PADDLES} TOF paddles')
    return sorted(paddle_ids)


def save_shard(graphs, output_path: Path, metadata: dict):
    temporary_path = output_path.with_suffix('.tmp')
    torch.save(graphs, temporary_path)
    temporary_path.replace(output_path)

    summary = {
        **metadata,
        'n_graphs': len(graphs),
        'size_mb': round(output_path.stat().st_size / 1024 / 1024, 2),
    }
    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    print(
        f'    saved {output_path.name}: {len(graphs):,} graphs, '
        f'{summary["size_mb"]:.1f} MB',
        flush=True,
    )
    return summary


def process_source(
    split: str,
    particle: str,
    source_index: int,
    pkl_path: Path,
    metadata_path: Path,
    output_dir: Path,
    builder: GraphBuilder,
    shard_size: int,
    min_hits: int,
    max_events: int | None,
):
    done_path = (
        output_dir / 'done'
        / f'{split}_{particle}_{source_index:03d}_{pkl_path.stem}.json'
    )
    if done_path.exists():
        with open(done_path, encoding='utf-8') as file:
            summary = json.load(file)
        print(
            f'  skip complete source: {pkl_path.name} '
            f'({summary["n_graphs"]:,} graphs)',
            flush=True,
        )
        return summary

    events = load_events(pkl_path)
    with np.load(metadata_path) as metadata:
        is_tracker_atrest = np.asarray(
            metadata['is_tracker_atrest'], dtype=np.bool_)
        event_id = np.asarray(metadata['event_id'], dtype=np.uint32)
        random_seed = np.asarray(metadata['random_seed'], dtype=np.uint32)

    if len(events) != len(is_tracker_atrest):
        raise RuntimeError(
            f'{pkl_path.name}: PKL/sidecar length mismatch '
            f'({len(events)} != {len(is_tracker_atrest)})')
    if len(event_id) != len(events):
        raise RuntimeError(
            f'{pkl_path.name}: event_id length mismatch')
    if len(random_seed) != len(events):
        raise RuntimeError(
            f'{pkl_path.name}: random_seed length mismatch')

    total = len(events)
    if max_events is not None:
        total = min(total, max_events)

    t0 = time.time()
    graphs = []
    shard_summaries = []
    label_counts = {0: 0, 1: 0}
    skipped_not_atrest = 0
    skipped_small = 0
    skipped_error = 0
    first_source_event_index = None
    last_source_event_index = None
    first_random_seed = None
    last_random_seed = None

    iterator = zip(
        events[:total],
        is_tracker_atrest[:total],
        event_id[:total],
        random_seed[:total],
    )
    iterator = tqdm(
        iterator,
        total=total,
        desc=f'{split} {source_index:03d}',
        dynamic_ncols=True,
    )
    for source_event_index, (
        event,
        keep_atrest,
        current_event_id,
        current_random_seed,
    ) in enumerate(iterator):
        if not keep_atrest:
            skipped_not_atrest += 1
            continue
        if int(event.get('n_hits', len(event['energy']))) <= min_hits:
            skipped_small += 1
            continue

        try:
            graph = builder.build_from_dict(event)
        except Exception as error:
            skipped_error += 1
            raise RuntimeError(
                f'{pkl_path.name}: graph build failed at '
                f'event_id={int(current_event_id)}: '
                f'{type(error).__name__}: {error}'
            ) from error

        current_event_id = int(current_event_id)
        current_random_seed = int(current_random_seed)
        graph.event_id = torch.tensor(
            [current_event_id], dtype=torch.long)
        graph.random_seed = torch.tensor(
            [current_random_seed], dtype=torch.long)
        graph.source_event_index = torch.tensor(
            [source_event_index], dtype=torch.long)
        if first_source_event_index is None:
            first_source_event_index = source_event_index
            first_random_seed = current_random_seed
        last_source_event_index = source_event_index
        last_random_seed = current_random_seed

        label = int(graph.y.item())
        label_counts[label] += 1
        graphs.append(graph)

        if len(graphs) >= shard_size:
            shard_index = len(shard_summaries)
            output_path = output_dir / (
                f'{split}_{particle}_{source_index:03d}_'
                f'{shard_index:02d}.pt'
            )
            shard_summaries.append(save_shard(
                graphs,
                output_path,
                {
                    'split': split,
                    'particle': particle,
                    'source_index': source_index,
                    'source_pkl': str(pkl_path),
                    'source_metadata': str(metadata_path),
                    'shard_index_within_source': shard_index,
                },
            ))
            graphs = []

    if graphs:
        shard_index = len(shard_summaries)
        output_path = output_dir / (
            f'{split}_{particle}_{source_index:03d}_{shard_index:02d}.pt'
        )
        shard_summaries.append(save_shard(
            graphs,
            output_path,
            {
                'split': split,
                'particle': particle,
                'source_index': source_index,
                'source_pkl': str(pkl_path),
                'source_metadata': str(metadata_path),
                'shard_index_within_source': shard_index,
            },
        ))

    n_graphs = sum(row['n_graphs'] for row in shard_summaries)
    if n_graphs == 0:
        raise RuntimeError(
            f'{pkl_path.name}: no graphs were produced; '
            'completion marker will not be written')

    summary = {
        'split': split,
        'particle': particle,
        'source_index': source_index,
        'source_pkl': str(pkl_path),
        'source_metadata': str(metadata_path),
        'source_events': total,
        'n_graphs': n_graphs,
        'n_shards': len(shard_summaries),
        'label_counts': {str(key): value for key, value in label_counts.items()},
        'skipped_not_atrest': skipped_not_atrest,
        'skipped_n_le_min_hits': skipped_small,
        'skipped_error': skipped_error,
        'first_saved_source_event_index': first_source_event_index,
        'last_saved_source_event_index': last_source_event_index,
        'first_saved_random_seed': first_random_seed,
        'last_saved_random_seed': last_random_seed,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_done = done_path.with_suffix('.tmp')
    with open(temporary_done, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    temporary_done.replace(done_path)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def load_manifest(path: Path) -> dict:
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def split_paths(
    manifest: dict,
    split: str,
    particles: list[str],
) -> list[tuple[str, Path]]:
    return [
        (particle, Path(path))
        for particle in particles
        for path in manifest[split][particle]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'val', 'test'],
        choices=['train', 'val', 'test'],
    )
    parser.add_argument(
        '--particles', nargs='+', default=['antiD', 'antiP'],
        choices=['antiD', 'antiP'],
    )
    parser.add_argument('--shard-size', type=int, default=50_000)
    parser.add_argument('--min-hits', type=int, default=8)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--max-events', type=int)
    parser.add_argument(
        '--paddle-ids',
        type=Path,
        help='reuse an existing paddle_ids.json instead of scanning train PKLs',
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.paddle_ids is not None:
        with open(args.paddle_ids, encoding='utf-8') as file:
            paddle_ids = [
                int(value) for value in json.load(file)['paddle_ids']
            ]
    else:
        print('collecting TOF paddle IDs from training PKLs...', flush=True)
        paddle_ids = collect_train_paddle_ids(
            [
                path for _, path in split_paths(
                    manifest, 'train', ['antiD', 'antiP'])
            ])

    if len(paddle_ids) != N_TOF_PADDLES:
        raise RuntimeError(
            f'expected {N_TOF_PADDLES} TOF paddles, got {len(paddle_ids)}')
    paddle_path = args.output_dir / 'paddle_ids.json'
    save_paddle_ids(paddle_ids, paddle_path)
    print(f'paddle mapping: {paddle_path}', flush=True)

    builder = GraphBuilder(
        k=args.k,
        normalize=True,
        tof_paddle_index=make_paddle_index(paddle_ids),
    )

    all_summaries = []
    for split in args.splits:
        print(f'\n=== {split} ===', flush=True)
        for particle in args.particles:
            paths = [
                path for _, path in split_paths(
                    manifest, split, [particle])
            ]
            if args.max_files is not None:
                paths = paths[:args.max_files]
            print(
                f'  {particle}: {len(paths)} source files',
                flush=True,
            )
            for source_index, pkl_path in enumerate(paths):
                metadata_path = sidecar_path(args.metadata_dir, pkl_path)
                if not pkl_path.exists():
                    raise FileNotFoundError(f'missing PKL: {pkl_path}')
                if not metadata_path.exists():
                    raise FileNotFoundError(
                        f'missing metadata sidecar: {metadata_path}')
                all_summaries.append(process_source(
                    split=split,
                    particle=particle,
                    source_index=source_index,
                    pkl_path=pkl_path,
                    metadata_path=metadata_path,
                    output_dir=args.output_dir,
                    builder=builder,
                    shard_size=args.shard_size,
                    min_hits=args.min_hits,
                    max_events=args.max_events,
                ))

    with open(
        args.output_dir / 'summary_all.json',
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(all_summaries, file, indent=2)

    total_graphs = sum(row['n_graphs'] for row in all_summaries)
    total_shards = sum(row['n_shards'] for row in all_summaries)
    print('\n========== complete ==========')
    print(f'sources: {len(all_summaries)}')
    print(f'graphs: {total_graphs:,}')
    print(f'shards: {total_shards:,}')
    print(f'output: {args.output_dir}')


if __name__ == '__main__':
    main()

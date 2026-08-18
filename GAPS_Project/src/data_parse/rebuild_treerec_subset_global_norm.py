"""Rebuild an existing TreeRec graph subset with train-set global normalization.

The input cache is an already selected split cache whose graphs retain
``random_seed`` and ``source_event_index``.  This script uses those fields to
retrieve exactly the same PKL events, so it changes feature representation only:
no event is added, removed, or moved between train/validation/test.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from GAPS_Project.src.data_parse.graph_builder import GraphBuilder


SPLITS = ('train', 'val', 'test')
# Keep provenance plus optional legacy TOF-paddle data.  The standard GravNet
# path does not consume tof_paddle_energy, but preserving it keeps rebuilt
# caches compatible with the GravNetTOF path as well.
COPY_FIELDS = (
    'event_id', 'random_seed', 'source_event_index', 'tof_paddle_energy',
)


def scalar(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def load_events(path: Path) -> list[dict]:
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    return payload['events'] if isinstance(payload, dict) and 'events' in payload else payload


def source_pkl_path(base_dir: Path, particle: str, seed: int) -> Path:
    return base_dir / particle / f'{particle}_2tof_FTFP_BERT_{seed}.pkl'


def source_info(graph) -> tuple[str, int, int]:
    if not hasattr(graph, 'random_seed') or not hasattr(graph, 'source_event_index'):
        raise RuntimeError('selected graph lacks random_seed/source_event_index metadata')
    particle = 'antiD' if scalar(graph.y) == 1 else 'antiP'
    return particle, scalar(graph.random_seed), scalar(graph.source_event_index)


def input_shards(cache_dir: Path, split: str) -> list[Path]:
    paths = sorted(cache_dir.glob(f'{split}_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no {split}_*.pt under {cache_dir}')
    return paths


def accumulate_train_statistics(
        cache_dir: Path, pkl_base_dir: Path, k: int,
        max_graphs_per_shard: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Return mean/std of [xyz, log1p(energy,time,dE/dx)] over selected train nodes."""
    builder = GraphBuilder(k=k, normalize=False)
    sum_x = np.zeros(6, dtype=np.float64)
    sum_x2 = np.zeros(6, dtype=np.float64)
    n_nodes = 0
    pkl_cache: dict[tuple[str, int], list[dict]] = {}

    for shard_path in input_shards(cache_dir, 'train'):
        graphs = torch.load(shard_path, map_location='cpu', weights_only=False)
        if max_graphs_per_shard is not None:
            graphs = graphs[:max_graphs_per_shard]
        for graph in graphs:
            particle, seed, source_index = source_info(graph)
            key = (particle, seed)
            if key not in pkl_cache:
                pkl_path = source_pkl_path(pkl_base_dir, particle, seed)
                if not pkl_path.exists():
                    raise FileNotFoundError(f'missing source PKL: {pkl_path}')
                pkl_cache[key] = load_events(pkl_path)
            event = pkl_cache[key][source_index]
            raw = builder.raw_node_features_from_dict(event)
            transformed = raw[:, :6].astype(np.float64, copy=True)
            transformed[:, 3:6] = np.log1p(np.clip(transformed[:, 3:6], 0.0, None))
            sum_x += transformed.sum(axis=0)
            sum_x2 += np.square(transformed).sum(axis=0)
            n_nodes += len(transformed)
        del graphs
        gc.collect()

    if n_nodes == 0:
        raise RuntimeError('no train nodes found while calculating statistics')
    mean = sum_x / n_nodes
    variance = np.maximum(sum_x2 / n_nodes - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), n_nodes


def rebuild_split(
        cache_dir: Path, pkl_base_dir: Path, output_dir: Path, split: str,
        builder: GraphBuilder, max_graphs_per_shard: int | None) -> dict:
    pkl_cache: dict[tuple[str, int], list[dict]] = {}
    total = 0
    label_counts: Counter[int] = Counter()

    for source_shard in input_shards(cache_dir, split):
        graphs = torch.load(source_shard, map_location='cpu', weights_only=False)
        if max_graphs_per_shard is not None:
            graphs = graphs[:max_graphs_per_shard]
        rebuilt = []
        for graph in graphs:
            particle, seed, source_index = source_info(graph)
            key = (particle, seed)
            if key not in pkl_cache:
                pkl_path = source_pkl_path(pkl_base_dir, particle, seed)
                if not pkl_path.exists():
                    raise FileNotFoundError(f'missing source PKL: {pkl_path}')
                pkl_cache[key] = load_events(pkl_path)
            event = pkl_cache[key][source_index]
            data = builder.build_from_dict(event)
            if scalar(data.y) != scalar(graph.y):
                raise RuntimeError(
                    f'label mismatch in {source_shard.name} at source event {source_index}')
            for field in COPY_FIELDS:
                if hasattr(graph, field):
                    setattr(data, field, getattr(graph, field))
            rebuilt.append(data)
            label_counts[scalar(data.y)] += 1

        destination = output_dir / source_shard.name
        if destination.exists():
            raise FileExistsError(f'refusing to overwrite {destination}')
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(rebuilt, temporary)
        temporary.replace(destination)
        with destination.with_suffix('.json').open('w', encoding='utf-8') as handle:
            json.dump({
                'n_graphs': len(rebuilt),
                'source_shard': str(source_shard.resolve()),
                'normalization': 'global_log',
            }, handle, indent=2)
        total += len(rebuilt)
        print(f'saved {destination.name}: {len(rebuilt):,} graphs', flush=True)
        del graphs, rebuilt
        gc.collect()
    return {'n_graphs': total, 'label_counts': dict(sorted(label_counts.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-cache-dir', type=Path, required=True)
    parser.add_argument('--pkl-base-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument(
        '--max-graphs-per-shard', type=int,
        help='small read/write smoke test only; omit for the exact full subset',
    )
    args = parser.parse_args()

    if args.max_graphs_per_shard is not None and args.max_graphs_per_shard < 1:
        raise ValueError('--max-graphs-per-shard must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mean, std, n_nodes = accumulate_train_statistics(
        args.source_cache_dir, args.pkl_base_dir, args.k,
        args.max_graphs_per_shard)
    print('train node statistics:', flush=True)
    for name, value_mean, value_std in zip(
            ('x', 'y', 'z', 'log1p_energy', 'log1p_time', 'log1p_dE_dx'),
            mean, std):
        print(f'  {name:14s} mean={value_mean:.7g} std={value_std:.7g}', flush=True)

    with (args.output_dir / 'node_feature_normalizer.json').open('w', encoding='utf-8') as handle:
        json.dump({
            'mode': 'global_log',
            'continuous_columns': [0, 1, 2, 3, 4, 5],
            'log1p_columns': [3, 4, 5],
            'unchanged_columns': {'6': 'det_type', '7': 'layer_norm'},
            'mean': mean.tolist(),
            'std': std.tolist(),
            'train_nodes': n_nodes,
            'source_cache_dir': str(args.source_cache_dir.resolve()),
            'max_graphs_per_shard': args.max_graphs_per_shard,
        }, handle, indent=2)

    builder = GraphBuilder(
        k=args.k,
        normalize=True,
        normalization_mode='global_log',
        global_feature_mean=mean,
        global_feature_std=std,
    )
    manifest = {
        'source_cache_dir': str(args.source_cache_dir.resolve()),
        'pkl_base_dir': str(args.pkl_base_dir.resolve()),
        'normalization': 'global_log',
        'max_graphs_per_shard': args.max_graphs_per_shard,
        'splits': {},
    }
    for split in SPLITS:
        manifest['splits'][split] = rebuild_split(
            args.source_cache_dir, args.pkl_base_dir, args.output_dir, split,
            builder, args.max_graphs_per_shard)
    with (args.output_dir / 'rebuild_manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    print('complete:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()

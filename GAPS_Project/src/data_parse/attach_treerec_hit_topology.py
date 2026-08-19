#!/usr/bin/env python3
"""Attach train-normalized, truth-free TreeRec topology features to graph shards.

The source graph cache is never modified.  A new cache is written so a
baseline and the topology-feature experiment can be compared on exactly the
same events and graph representation.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from GAPS_Project.src.data_parse.treerec_hit_topology import (
    STABLE_FEATURE_NAMES,
    load_node_normalizer,
    stable_topology_features,
)


SPLITS = ('train', 'val', 'test')


def shard_paths(cache_dir: Path, split: str) -> list[Path]:
    paths = sorted(cache_dir.glob(f'{split}_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no {split}_*.pt shards under {cache_dir}')
    return paths


def selected_graphs(path: Path, max_graphs_per_shard: int | None):
    graphs = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(graphs, list) or not graphs:
        raise RuntimeError(f'invalid graph shard: {path}')
    return graphs[:max_graphs_per_shard] if max_graphs_per_shard is not None else graphs


def fit_train_normalizer(
        cache_dir: Path, node_mean: np.ndarray, node_std: np.ndarray,
        max_graphs_per_shard: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    sum_x = np.zeros(len(STABLE_FEATURE_NAMES), dtype=np.float64)
    sum_x2 = np.zeros_like(sum_x)
    n_graphs = 0
    for index, path in enumerate(shard_paths(cache_dir, 'train'), start=1):
        graphs = selected_graphs(path, max_graphs_per_shard)
        for graph in graphs:
            value = stable_topology_features(graph, node_mean, node_std)
            sum_x += value
            sum_x2 += np.square(value)
            n_graphs += 1
        print(f'[train stats {index:02d}] {path.name}: {len(graphs):,} events', flush=True)
        del graphs
        gc.collect()
    mean = sum_x / n_graphs
    variance = np.maximum(sum_x2 / n_graphs - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).clip(1e-6).astype(np.float32), n_graphs


def rebuild_split(
        source_dir: Path, output_dir: Path, split: str,
        node_mean: np.ndarray, node_std: np.ndarray,
        topology_mean: np.ndarray, topology_std: np.ndarray,
        max_graphs_per_shard: int | None) -> dict:
    labels: Counter[int] = Counter()
    total = 0
    for index, source in enumerate(shard_paths(source_dir, split), start=1):
        graphs = selected_graphs(source, max_graphs_per_shard)
        for graph in graphs:
            raw = stable_topology_features(graph, node_mean, node_std)
            graph.hit_topology = torch.tensor(raw, dtype=torch.float32)
            graph.hit_topology_z = torch.tensor(
                (raw - topology_mean) / topology_std, dtype=torch.float32)
            labels[int(graph.y.reshape(-1)[0].item())] += 1

        destination = output_dir / source.name
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(graphs, temporary)
        temporary.replace(destination)
        destination.with_suffix('.json').write_text(json.dumps({
            'n_graphs': len(graphs),
            'source_shard': str(source.resolve()),
            'hit_topology_feature_names': list(STABLE_FEATURE_NAMES),
            'normalization': 'train_global_zscore',
        }, indent=2), encoding='utf-8')
        total += len(graphs)
        print(f'[{split} {index:02d}] saved {destination.name}: {len(graphs):,} events', flush=True)
        del graphs
        gc.collect()
    return {'n_graphs': total, 'label_counts': dict(sorted(labels.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--node-normalizer', type=Path)
    parser.add_argument('--max-graphs-per-shard', type=int,
                        help='small smoke cache only; omit for the exact 1M cache')
    args = parser.parse_args()

    if args.max_graphs_per_shard is not None and args.max_graphs_per_shard < 1:
        raise ValueError('--max-graphs-per-shard must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    node_normalizer = args.node_normalizer or args.source_cache_dir / 'node_feature_normalizer.json'
    node_mean, node_std = load_node_normalizer(node_normalizer)
    topology_mean, topology_std, train_events = fit_train_normalizer(
        args.source_cache_dir, node_mean, node_std, args.max_graphs_per_shard)
    print('topology train statistics:', flush=True)
    for name, mean, std in zip(STABLE_FEATURE_NAMES, topology_mean, topology_std):
        print(f'  {name:38s} mean={mean:.7g} std={std:.7g}', flush=True)

    normalizer = {
        'feature_names': list(STABLE_FEATURE_NAMES),
        'mean': topology_mean.tolist(),
        'std': topology_std.tolist(),
        'train_events': train_events,
        'source_cache_dir': str(args.source_cache_dir.resolve()),
        'node_normalizer': str(node_normalizer.resolve()),
        'max_graphs_per_shard': args.max_graphs_per_shard,
    }
    (args.output_dir / 'hit_topology_normalizer.json').write_text(
        json.dumps(normalizer, indent=2), encoding='utf-8')
    shutil.copy2(node_normalizer, args.output_dir / 'node_feature_normalizer.json')

    manifest = {
        'source_cache_dir': str(args.source_cache_dir.resolve()),
        'hit_topology_feature_names': list(STABLE_FEATURE_NAMES),
        'max_graphs_per_shard': args.max_graphs_per_shard,
        'splits': {},
    }
    for split in SPLITS:
        manifest['splits'][split] = rebuild_split(
            args.source_cache_dir, args.output_dir, split,
            node_mean, node_std, topology_mean, topology_std,
            args.max_graphs_per_shard)
    (args.output_dir / 'topology_cache_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'complete: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()

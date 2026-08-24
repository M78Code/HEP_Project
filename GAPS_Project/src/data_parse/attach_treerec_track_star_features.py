#!/usr/bin/env python3
"""Attach train-normalized TreeRec track/star geometry candidates to a cache.

The source cache remains untouched.  This provides a strict same-events A/B
against the global-log baseline, using only four non-degenerate hit geometry
features selected by the independent audit.
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

from GAPS_Project.src.data_parse.treerec_hit_topology import load_node_normalizer
from GAPS_Project.src.data_parse.treerec_track_star import (
    STRUCTURAL_FEATURE_NAMES,
    structural_track_star_features,
)


SPLITS = ('train', 'val', 'test')


def shard_paths(cache_dir: Path, split: str) -> list[Path]:
    paths = sorted(cache_dir.glob(f'{split}_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no {split}_*.pt shards under {cache_dir}')
    return paths


def load_graphs(path: Path, limit: int | None) -> list:
    graphs = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(graphs, list) or not graphs:
        raise RuntimeError(f'invalid graph shard: {path}')
    return graphs[:limit] if limit is not None else graphs


def fit_normalizer(
        cache_dir: Path, node_mean: np.ndarray, node_std: np.ndarray,
        limit: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(len(STRUCTURAL_FEATURE_NAMES), dtype=np.float64)
    total_sq = np.zeros_like(total)
    events = 0
    for index, path in enumerate(shard_paths(cache_dir, 'train'), start=1):
        graphs = load_graphs(path, limit)
        for graph in graphs:
            raw = structural_track_star_features(graph, node_mean, node_std)
            total += raw
            total_sq += raw * raw
            events += 1
        print(f'[train stats {index:02d}] {path.name}: {len(graphs):,} events', flush=True)
        del graphs
        gc.collect()
    mean = total / events
    variance = np.maximum(total_sq / events - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).clip(1e-6).astype(np.float32), events


def write_split(
        source_dir: Path, output_dir: Path, split: str,
        node_mean: np.ndarray, node_std: np.ndarray,
        feature_mean: np.ndarray, feature_std: np.ndarray,
        limit: int | None) -> dict:
    labels: Counter[int] = Counter()
    events = 0
    for index, source in enumerate(shard_paths(source_dir, split), start=1):
        graphs = load_graphs(source, limit)
        for graph in graphs:
            raw = structural_track_star_features(graph, node_mean, node_std)
            graph.track_star = torch.tensor(raw, dtype=torch.float32)
            graph.track_star_z = torch.tensor(
                (raw - feature_mean) / feature_std, dtype=torch.float32)
            labels[int(graph.y.reshape(-1)[0].item())] += 1
        destination = output_dir / source.name
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(graphs, temporary)
        temporary.replace(destination)
        destination.with_suffix('.json').write_text(json.dumps({
            'n_graphs': len(graphs),
            'source_shard': str(source.resolve()),
            'track_star_feature_names': list(STRUCTURAL_FEATURE_NAMES),
            'normalization': 'train_global_zscore',
        }, indent=2), encoding='utf-8')
        events += len(graphs)
        print(f'[{split} {index:02d}] saved {destination.name}: {len(graphs):,} events', flush=True)
        del graphs
        gc.collect()
    return {'n_graphs': events, 'label_counts': dict(sorted(labels.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--node-normalizer', type=Path)
    parser.add_argument('--max-graphs-per-shard', type=int)
    args = parser.parse_args()

    if args.max_graphs_per_shard is not None and args.max_graphs_per_shard < 1:
        raise ValueError('--max-graphs-per-shard must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    node_normalizer = args.node_normalizer or args.source_cache_dir / 'node_feature_normalizer.json'
    node_mean, node_std = load_node_normalizer(node_normalizer)
    feature_mean, feature_std, train_events = fit_normalizer(
        args.source_cache_dir, node_mean, node_std, args.max_graphs_per_shard)
    print('track/star train statistics:', flush=True)
    for name, mean, std in zip(STRUCTURAL_FEATURE_NAMES, feature_mean, feature_std):
        print(f'  {name:52s} mean={mean:.7g} std={std:.7g}', flush=True)

    normalizer = {
        'feature_names': list(STRUCTURAL_FEATURE_NAMES),
        'mean': feature_mean.tolist(),
        'std': feature_std.tolist(),
        'train_events': train_events,
        'source_cache_dir': str(args.source_cache_dir.resolve()),
        'node_normalizer': str(node_normalizer.resolve()),
        'max_graphs_per_shard': args.max_graphs_per_shard,
    }
    (args.output_dir / 'track_star_normalizer.json').write_text(
        json.dumps(normalizer, indent=2), encoding='utf-8')
    shutil.copy2(node_normalizer, args.output_dir / 'node_feature_normalizer.json')

    manifest = {
        'source_cache_dir': str(args.source_cache_dir.resolve()),
        'track_star_feature_names': list(STRUCTURAL_FEATURE_NAMES),
        'max_graphs_per_shard': args.max_graphs_per_shard,
        'splits': {},
    }
    for split in SPLITS:
        manifest['splits'][split] = write_split(
            args.source_cache_dir, args.output_dir, split,
            node_mean, node_std, feature_mean, feature_std,
            args.max_graphs_per_shard)
    (args.output_dir / 'track_star_cache_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'complete: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Attach normalized TreeRec vertex/prong tokens to graph-cache shards.

The source cache is read-only.  Tokens are derived solely from observed
TreeRec Si(Li) hit positions and energies: one local energy-density vertex
candidate and up to four spatially connected outer-prong candidates.
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

from GAPS_Project.src.data_parse.treerec_cluster_vertex import (
    MAX_OUTER_PRONGS,
    PRONG_TOKEN_DIM,
    VERTEX_TOKEN_DIM,
    cluster_vertex_tokens,
)
from GAPS_Project.src.data_parse.treerec_hit_topology import load_node_normalizer


SPLITS = ('train', 'val', 'test')
VERTEX_TOKEN_NAMES = ('has_vertex', 'vertex_energy_fraction_75mm',
                      'vertex_energy_fraction_125mm')
PRONG_TOKEN_NAMES = ('energy_fraction', 'log_distance_mm', 'direction_x',
                     'direction_y', 'direction_z', 'log_hit_count',
                     'log_rms_radius_mm')


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
        limit: int | None, max_outer_prongs: int) -> tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    vertex_sum = np.zeros(VERTEX_TOKEN_DIM, dtype=np.float64)
    vertex_sum_sq = np.zeros_like(vertex_sum)
    prong_sum = np.zeros(PRONG_TOKEN_DIM, dtype=np.float64)
    prong_sum_sq = np.zeros_like(prong_sum)
    events = 0
    active_prongs = 0
    for index, path in enumerate(shard_paths(cache_dir, 'train'), start=1):
        graphs = load_graphs(path, limit)
        for graph in graphs:
            vertex, prongs, mask = cluster_vertex_tokens(
                graph, node_mean, node_std, max_outer_prongs)
            vertex_sum += vertex
            vertex_sum_sq += np.square(vertex)
            if mask.any():
                values = prongs[mask]
                prong_sum += values.sum(axis=0)
                prong_sum_sq += np.square(values).sum(axis=0)
                active_prongs += len(values)
            events += 1
        print(f'[train stats {index:02d}] {path.name}: {len(graphs):,} events', flush=True)
        del graphs
        gc.collect()
    if events == 0 or active_prongs == 0:
        raise ValueError('training cache does not contain enough cluster/prong tokens')
    vertex_mean = vertex_sum / events
    vertex_std = np.sqrt(np.maximum(vertex_sum_sq / events - vertex_mean ** 2, 0.0))
    prong_mean = prong_sum / active_prongs
    prong_std = np.sqrt(np.maximum(prong_sum_sq / active_prongs - prong_mean ** 2, 0.0))
    return (
        vertex_mean.astype(np.float32), vertex_std.clip(1e-6).astype(np.float32),
        prong_mean.astype(np.float32), prong_std.clip(1e-6).astype(np.float32),
        events, active_prongs,
    )


def write_split(
        source_dir: Path, output_dir: Path, split: str,
        node_mean: np.ndarray, node_std: np.ndarray,
        vertex_mean: np.ndarray, vertex_std: np.ndarray,
        prong_mean: np.ndarray, prong_std: np.ndarray,
        limit: int | None, max_outer_prongs: int) -> dict:
    labels: Counter[int] = Counter()
    events = 0
    active_prongs = 0
    for index, source in enumerate(shard_paths(source_dir, split), start=1):
        graphs = load_graphs(source, limit)
        for graph in graphs:
            vertex, prongs, mask = cluster_vertex_tokens(
                graph, node_mean, node_std, max_outer_prongs)
            vertex_z = (vertex - vertex_mean) / vertex_std
            prongs_z = np.zeros_like(prongs)
            if mask.any():
                prongs_z[mask] = (prongs[mask] - prong_mean) / prong_std
                active_prongs += int(mask.sum())
            graph.cluster_vertex_token = torch.tensor(vertex, dtype=torch.float32)
            graph.cluster_vertex_token_z = torch.tensor(vertex_z, dtype=torch.float32)
            graph.cluster_prong_tokens = torch.tensor(prongs, dtype=torch.float32)
            graph.cluster_prong_tokens_z = torch.tensor(prongs_z, dtype=torch.float32)
            graph.cluster_prong_mask = torch.tensor(mask, dtype=torch.bool)
            labels[int(graph.y.reshape(-1)[0].item())] += 1
        destination = output_dir / source.name
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(graphs, temporary)
        temporary.replace(destination)
        destination.with_suffix('.json').write_text(json.dumps({
            'n_graphs': len(graphs),
            'source_shard': str(source.resolve()),
            'cluster_vertex_token_dim': VERTEX_TOKEN_DIM,
            'cluster_prong_token_dim': PRONG_TOKEN_DIM,
            'max_outer_prongs': max_outer_prongs,
            'normalization': 'train_global_zscore_active_prongs_only',
        }, indent=2), encoding='utf-8')
        events += len(graphs)
        print(f'[{split} {index:02d}] saved {destination.name}: {len(graphs):,} events', flush=True)
        del graphs
        gc.collect()
    return {
        'n_graphs': events,
        'active_prongs': active_prongs,
        'label_counts': dict(sorted(labels.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--node-normalizer', type=Path)
    parser.add_argument('--max-graphs-per-shard', type=int,
                        help='small smoke cache only; omit for the exact 1M cache')
    parser.add_argument('--max-outer-prongs', type=int, default=MAX_OUTER_PRONGS)
    args = parser.parse_args()

    if args.max_graphs_per_shard is not None and args.max_graphs_per_shard < 1:
        raise ValueError('--max-graphs-per-shard must be positive')
    if args.max_outer_prongs < 1:
        raise ValueError('--max-outer-prongs must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    node_normalizer = args.node_normalizer or args.source_cache_dir / 'node_feature_normalizer.json'
    node_mean, node_std = load_node_normalizer(node_normalizer)
    vertex_mean, vertex_std, prong_mean, prong_std, train_events, active_prongs = fit_normalizer(
        args.source_cache_dir, node_mean, node_std,
        args.max_graphs_per_shard, args.max_outer_prongs)
    print('cluster/vertex token train statistics:', flush=True)
    for name, mean, std in zip(VERTEX_TOKEN_NAMES, vertex_mean, vertex_std):
        print(f'  vertex {name:39s} mean={mean:.7g} std={std:.7g}', flush=True)
    for name, mean, std in zip(PRONG_TOKEN_NAMES, prong_mean, prong_std):
        print(f'  prong  {name:39s} mean={mean:.7g} std={std:.7g}', flush=True)
    print(f'  active prongs in train: {active_prongs:,}', flush=True)

    normalizer = {
        'vertex_token_names': list(VERTEX_TOKEN_NAMES),
        'prong_token_names': list(PRONG_TOKEN_NAMES),
        'vertex_mean': vertex_mean.tolist(), 'vertex_std': vertex_std.tolist(),
        'prong_mean': prong_mean.tolist(), 'prong_std': prong_std.tolist(),
        'train_events': train_events, 'train_active_prongs': active_prongs,
        'source_cache_dir': str(args.source_cache_dir.resolve()),
        'node_normalizer': str(node_normalizer.resolve()),
        'max_outer_prongs': args.max_outer_prongs,
    }
    (args.output_dir / 'cluster_vertex_token_normalizer.json').write_text(
        json.dumps(normalizer, indent=2), encoding='utf-8')
    shutil.copy2(node_normalizer, args.output_dir / 'node_feature_normalizer.json')

    manifest = {
        'source_cache_dir': str(args.source_cache_dir.resolve()),
        'max_graphs_per_shard': args.max_graphs_per_shard,
        'max_outer_prongs': args.max_outer_prongs,
        'splits': {},
    }
    for split in SPLITS:
        manifest['splits'][split] = write_split(
            args.source_cache_dir, args.output_dir, split,
            node_mean, node_std, vertex_mean, vertex_std, prong_mean, prong_std,
            args.max_graphs_per_shard, args.max_outer_prongs)
    (args.output_dir / 'cluster_vertex_token_cache_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'complete: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()

"""Audit the physical kNN edges stored in a TreeRec graph cache.

The current GravNet baseline ignores ``edge_index``.  This script checks that
the cached edges are usable as a separate, explicit physics-relation branch:
their multiplicity, geometric length, and detector-pair composition.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PAIR_NAMES = {
    (0, 0): 'TOF->TOF',
    (0, 1): 'TOF->SiLi',
    (1, 0): 'SiLi->TOF',
    (1, 1): 'SiLi->SiLi',
}


def percentile_summary(values: list[float]) -> str:
    if not values:
        return 'n=0'
    array = np.asarray(values, dtype=np.float64)
    p50, p90, p99 = np.percentile(array, [50, 90, 99])
    return (
        f'n={len(array):,} mean={array.mean():.3f} '
        f'p50={p50:.3f} p90={p90:.3f} p99={p99:.3f} max={array.max():.3f}')


def split_paths(cache_dir: Path, split: str) -> list[Path]:
    paths = sorted(cache_dir.glob(f'{split}_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no {split}_*.pt files under {cache_dir}')
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['test'],
                        choices=['train', 'val', 'test'])
    parser.add_argument('--max-events', type=int, default=10_000)
    parser.add_argument('--max-events-per-file', type=int)
    args = parser.parse_args()

    if args.max_events < 1:
        raise ValueError('--max-events must be positive')
    if args.max_events_per_file is not None and args.max_events_per_file < 1:
        raise ValueError('--max-events-per-file must be positive')

    total_events = 0
    nodes_per_event: list[float] = []
    edges_per_event: list[float] = []
    lengths_by_pair = {pair: [] for pair in PAIR_NAMES}
    pair_counts: Counter[tuple[int, int]] = Counter()
    bad_events = 0

    for split in args.splits:
        for path in split_paths(args.cache_dir, split):
            if total_events >= args.max_events:
                break
            graphs = torch.load(path, map_location='cpu', weights_only=False)
            if args.max_events_per_file is not None:
                graphs = graphs[:args.max_events_per_file]
            for graph in graphs:
                if total_events >= args.max_events:
                    break
                if not hasattr(graph, 'pos') or not hasattr(graph, 'edge_index'):
                    raise RuntimeError(f'{path} is missing pos or edge_index')
                if graph.x.ndim != 2 or graph.x.size(1) < 7:
                    raise RuntimeError(f'{path} does not contain 8-D TreeRec node features')

                pos = graph.pos.float()
                edge_index = graph.edge_index.long()
                n_nodes = int(pos.size(0))
                n_edges = int(edge_index.size(1))
                if edge_index.ndim != 2 or edge_index.size(0) != 2:
                    raise RuntimeError(f'invalid edge_index shape in {path}: {tuple(edge_index.shape)}')
                if n_edges and (edge_index.min() < 0 or edge_index.max() >= n_nodes):
                    raise RuntimeError(f'out-of-range edge endpoint in {path}')

                nodes_per_event.append(float(n_nodes))
                edges_per_event.append(float(n_edges))
                if n_edges:
                    source, target = edge_index
                    distances = (pos[source] - pos[target]).norm(dim=1).numpy()
                    detector = graph.x[:, 6].round().long()
                    source_detector = detector[source].numpy()
                    target_detector = detector[target].numpy()
                    for src_det, dst_det, distance in zip(
                            source_detector, target_detector, distances):
                        pair = (int(src_det), int(dst_det))
                        if pair not in PAIR_NAMES:
                            raise RuntimeError(f'unexpected detector encoding {pair} in {path}')
                        pair_counts[pair] += 1
                        lengths_by_pair[pair].append(float(distance))
                else:
                    bad_events += 1
                total_events += 1
            print(f'[{split}] {path.name}: cumulative events={total_events:,}', flush=True)
        if total_events >= args.max_events:
            break

    total_edges = sum(pair_counts.values())
    print('\nTreeRec physical-edge audit')
    print(f'events audited: {total_events:,}')
    print(f'events with zero edges: {bad_events:,}')
    print(f'nodes/event: {percentile_summary(nodes_per_event)}')
    print(f'edges/event: {percentile_summary(edges_per_event)}')
    print(f'total directed edges: {total_edges:,}')
    for pair, name in PAIR_NAMES.items():
        count = pair_counts[pair]
        fraction = count / total_edges if total_edges else 0.0
        print(f'{name:12s} fraction={fraction:.4%} {percentile_summary(lengths_by_pair[pair])}')


if __name__ == '__main__':
    main()

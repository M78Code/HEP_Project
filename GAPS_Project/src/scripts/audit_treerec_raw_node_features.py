"""Audit raw TreeRec node features for an already selected graph-cache subset.

Each cached graph keeps ``random_seed`` and ``source_event_index``.  Those
fields let this tool retrieve the exact source PKL event and rebuild its node
features with ``normalize=False``.  It is intentionally read-only: use it to
choose a replacement normalization policy before generating another cache.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from GAPS_Project.src.data_parse.graph_builder import GraphBuilder


FEATURE_NAMES = [
    'x_mm', 'y_mm', 'z_mm', 'energy', 'time_ns', 'dE_dx', 'det_type', 'layer_norm',
]


def load_events(path: Path) -> list[dict]:
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    return payload['events'] if isinstance(payload, dict) and 'events' in payload else payload


def source_pkl_path(base_dir: Path, particle: str, random_seed: int) -> Path:
    return base_dir / particle / f'{particle}_2tof_FTFP_BERT_{random_seed}.pkl'


def scalar(value: torch.Tensor) -> int:
    return int(value.reshape(-1)[0].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--pkl-base-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--samples-per-shard', type=int, default=100)
    parser.add_argument('--k', type=int, default=8)
    args = parser.parse_args()

    paths = sorted(args.cache_dir.glob(f'{args.split}_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no {args.split}_*.pt under {args.cache_dir}')
    if args.samples_per_shard < 1:
        raise ValueError('--samples-per-shard must be positive')

    builder = GraphBuilder(k=args.k, normalize=False)
    raw_nodes: list[np.ndarray] = []
    cached_nodes: list[np.ndarray] = []
    labels = Counter()
    raw_detector_types = Counter()
    checked = 0
    mapping_errors: list[str] = []

    for shard_path in paths:
        graphs = torch.load(shard_path, map_location='cpu', weights_only=False)
        take = min(args.samples_per_shard, len(graphs))
        indices = np.linspace(0, len(graphs) - 1, take, dtype=np.int64)

        selected: dict[tuple[str, int], list[tuple[object, int]]] = {}
        for index in indices:
            graph = graphs[int(index)]
            particle = 'antiD' if scalar(graph.y) == 1 else 'antiP'
            if not hasattr(graph, 'random_seed') or not hasattr(graph, 'source_event_index'):
                raise RuntimeError(
                    f'{shard_path.name} has no random_seed/source_event_index metadata')
            seed = scalar(graph.random_seed)
            selected.setdefault((particle, seed), []).append(
                (graph, scalar(graph.source_event_index)))

        for (particle, seed), entries in selected.items():
            pkl_path = source_pkl_path(args.pkl_base_dir, particle, seed)
            if not pkl_path.exists():
                mapping_errors.append(f'missing source PKL: {pkl_path}')
                continue
            events = load_events(pkl_path)
            for graph, source_index in entries:
                if source_index >= len(events):
                    mapping_errors.append(
                        f'{pkl_path.name}: source index {source_index} >= {len(events)}')
                    continue
                raw_graph = builder.build_from_dict(events[source_index])
                if scalar(raw_graph.y) != scalar(graph.y):
                    mapping_errors.append(
                        f'{pkl_path.name}:{source_index}: label mismatch')
                    continue
                raw_x = raw_graph.x.numpy()
                raw_nodes.append(raw_x)
                cached_nodes.append(graph.x.numpy())
                labels[scalar(graph.y)] += 1
                raw_detector_types.update(raw_x[:, 6].astype(np.int64).tolist())
                checked += 1

    print(f'shards: {len(paths)}')
    print(f'graphs checked: {checked}')
    print(f'labels: {dict(labels)}')
    print(f'raw detector types: {dict(sorted(raw_detector_types.items()))}')
    print(f'mapping errors: {len(mapping_errors)}')
    for error in mapping_errors[:10]:
        print(f'  {error}')
    if mapping_errors:
        raise RuntimeError('source mapping audit failed')
    if not raw_nodes:
        raise RuntimeError('no nodes audited')

    raw = np.concatenate(raw_nodes, axis=0)
    cached = np.concatenate(cached_nodes, axis=0)
    quantiles = [0.0, 0.01, 0.50, 0.99, 1.0]
    print(f'nodes audited: {len(raw):,}')
    print('\nraw node feature quantiles')
    for column, name in enumerate(FEATURE_NAMES):
        values = np.quantile(raw[:, column], quantiles)
        print(
            f'{name:12s} '
            f'q00={values[0]:.6g} q01={values[1]:.6g} '
            f'q50={values[2]:.6g} q99={values[3]:.6g} q100={values[4]:.6g}'
        )

    print('\ncached node feature global summary')
    for column, name in enumerate(FEATURE_NAMES):
        print(
            f'{name:12s} mean={cached[:, column].mean():.6g} '
            f'std={cached[:, column].std():.6g} '
            f'min={cached[:, column].min():.6g} max={cached[:, column].max():.6g}'
        )


if __name__ == '__main__':
    main()

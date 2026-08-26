"""Attach train-normalized physical attributes to cached TreeRec kNN edges.

The source graph cache already stores position-based directed kNN ``edge_index``.
This script preserves every graph and adds only ``edge_attr``.  Continuous edge
features are standardized using train edges only; detector/layer relation bits
remain categorical.  No labels, TreeMc fields, or reconstructed-track products
are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


SPLITS = ('train', 'val', 'test')
CONTINUOUS_NAMES = (
    'dx_mm', 'dy_mm', 'dz_mm', 'log1p_distance_mm',
    'delta_log_time_z', 'delta_log_energy_z', 'delta_log_dedx_z',
)
CATEGORICAL_NAMES = ('source_is_sili', 'target_is_sili', 'same_layer')


def split_paths(cache_dir: Path, split: str) -> list[Path]:
    paths = sorted(cache_dir.glob(f'{split}_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no {split}_*.pt files under {cache_dir}')
    return paths


def limited(graphs: list, max_graphs_per_shard: int | None) -> list:
    if max_graphs_per_shard is None:
        return graphs
    return graphs[:max_graphs_per_shard]


def raw_edge_features(graph) -> tuple[torch.Tensor, torch.Tensor]:
    """Return continuous and categorical attributes for directed cached edges."""
    if not hasattr(graph, 'pos') or not hasattr(graph, 'edge_index'):
        raise RuntimeError('graph is missing pos or edge_index')
    if graph.x.ndim != 2 or graph.x.size(1) < 8:
        raise RuntimeError('expected TreeRec 8-D node features')

    source, target = graph.edge_index.long()
    pos = graph.pos.float()
    delta_pos = pos[target] - pos[source]
    log_distance = torch.log1p(delta_pos.norm(dim=1, keepdim=True))

    # In global-log caches x[:, 3:6] are train-global standardized
    # [log1p(energy), log1p(time), log1p(dE/dx)].  Differences preserve the
    # hit-to-hit ordering and are standardized again at edge level below.
    delta_log_time = (graph.x[target, 4] - graph.x[source, 4]).float().unsqueeze(1)
    delta_log_energy = (graph.x[target, 3] - graph.x[source, 3]).float().unsqueeze(1)
    delta_log_dedx = (graph.x[target, 5] - graph.x[source, 5]).float().unsqueeze(1)
    continuous = torch.cat(
        [delta_pos, log_distance, delta_log_time, delta_log_energy, delta_log_dedx],
        dim=1,
    )

    detector = graph.x[:, 6].round().long()
    layer = graph.x[:, 7]
    categorical = torch.stack([
        (detector[source] == 1).float(),
        (detector[target] == 1).float(),
        torch.isclose(layer[source], layer[target]).float(),
    ], dim=1)
    if not torch.isfinite(continuous).all() or not torch.isfinite(categorical).all():
        raise RuntimeError('non-finite physics edge feature encountered')
    return continuous, categorical


def fit_train_normalizer(
        source_dir: Path, max_graphs_per_shard: int | None) -> tuple[torch.Tensor, torch.Tensor, int]:
    total = torch.zeros(len(CONTINUOUS_NAMES), dtype=torch.float64)
    total_squared = torch.zeros(len(CONTINUOUS_NAMES), dtype=torch.float64)
    n_edges = 0
    for shard_number, path in enumerate(split_paths(source_dir, 'train'), start=1):
        graphs = limited(
            torch.load(path, map_location='cpu', weights_only=False),
            max_graphs_per_shard,
        )
        for graph in graphs:
            continuous, _ = raw_edge_features(graph)
            total += continuous.double().sum(dim=0)
            total_squared += continuous.double().square().sum(dim=0)
            n_edges += continuous.size(0)
        print(
            f'[train stats {shard_number:02d}] {path.name}: '
            f'{len(graphs):,} events, cumulative edges={n_edges:,}',
            flush=True,
        )
    if n_edges == 0:
        raise RuntimeError('no train edges found')
    mean = total / n_edges
    variance = (total_squared / n_edges - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(1e-6)
    return mean.float(), std.float(), n_edges


def attach_split(
        source_dir: Path, output_dir: Path, split: str,
        mean: torch.Tensor, std: torch.Tensor,
        max_graphs_per_shard: int | None) -> None:
    for shard_number, source_path in enumerate(split_paths(source_dir, split), start=1):
        destination = output_dir / source_path.name
        if destination.exists():
            raise FileExistsError(f'refusing to overwrite {destination}')
        graphs = limited(
            torch.load(source_path, map_location='cpu', weights_only=False),
            max_graphs_per_shard,
        )
        for graph in graphs:
            continuous, categorical = raw_edge_features(graph)
            graph.edge_attr = torch.cat([(continuous - mean) / std, categorical], dim=1)
        torch.save(graphs, destination)
        with destination.with_suffix('.json').open('w', encoding='utf-8') as handle:
            json.dump({
                'n_graphs': len(graphs),
                'source_shard': str(source_path.resolve()),
                'edge_attr_dim': len(CONTINUOUS_NAMES) + len(CATEGORICAL_NAMES),
            }, handle, indent=2)
        print(f'[{split} {shard_number:02d}] saved {destination.name}: {len(graphs):,} events', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-graphs-per-shard', type=int,
                        help='small smoke cache only; omit for exact full cache')
    args = parser.parse_args()
    if args.max_graphs_per_shard is not None and args.max_graphs_per_shard < 1:
        raise ValueError('--max-graphs-per-shard must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mean, std, n_edges = fit_train_normalizer(
        args.source_cache_dir, args.max_graphs_per_shard)
    print('physics edge train statistics:')
    for name, value_mean, value_std in zip(CONTINUOUS_NAMES, mean, std):
        print(f'  {name:28s} mean={value_mean.item():.7g} std={value_std.item():.7g}')
    with (args.output_dir / 'physics_edge_normalizer.json').open('w', encoding='utf-8') as handle:
        json.dump({
            'continuous_names': list(CONTINUOUS_NAMES),
            'categorical_names': list(CATEGORICAL_NAMES),
            'continuous_mean': mean.tolist(),
            'continuous_std': std.tolist(),
            'train_edges': n_edges,
        }, handle, indent=2)

    for split in SPLITS:
        attach_split(
            args.source_cache_dir, args.output_dir, split, mean, std,
            args.max_graphs_per_shard)
    print(f'complete: {args.output_dir}')


if __name__ == '__main__':
    main()

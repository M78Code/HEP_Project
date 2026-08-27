"""Attach training-only MC stop/direction targets to a TreeRec graph cache.

The model input remains the original TreeRec graph.  The added fields are
consumed only by the training loss of ``gravnet_soft_objects`` and are never
passed into its forward method.  Events are joined to TreeMc by the persisted
event_id provenance field, rather than by TreeRec hit indices.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot


SPLITS = ('train', 'val', 'test')
MC_BRANCHES = (
    'Mc/CEventBase/eventId_',
    'Mc/primaryStoppingPosition_',
    'Mc/CEventBase/primaryMomentumDirectionGenerated_',
)


def scalar(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def vector_array(values) -> np.ndarray:
    return np.column_stack([
        ak.to_numpy(values['fX']),
        ak.to_numpy(values['fY']),
        ak.to_numpy(values['fZ']),
    ]).astype(np.float32, copy=False)


def input_shards(cache_dir: Path, split: str) -> list[Path]:
    paths = sorted(cache_dir.glob(f'{split}_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no {split}_*.pt under {cache_dir}')
    return paths


class TruthLookup:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.cache: dict[tuple[str, int], dict[int, tuple[np.ndarray, np.ndarray, bool]]] = {}

    def _load(self, particle: str, random_seed: int) -> None:
        key = (particle, random_seed)
        if key in self.cache:
            return
        path = self.root_dir / particle / (
            f'{particle}_2tof_FTFP_BERT_{random_seed}.root')
        if not path.exists():
            raise FileNotFoundError(f'missing source ROOT: {path}')
        with uproot.open(path) as root_file:
            mc = root_file['TreeMc']
            arrays = mc.arrays(MC_BRANCHES, library='ak')
        event_ids = ak.to_numpy(arrays['Mc/CEventBase/eventId_']).astype(np.int64)
        stops = vector_array(arrays['Mc/primaryStoppingPosition_'])
        directions = vector_array(
            arrays['Mc/CEventBase/primaryMomentumDirectionGenerated_'])
        if not (len(event_ids) == len(stops) == len(directions)):
            raise RuntimeError(f'{path}: inconsistent TreeMc target lengths')
        if len(np.unique(event_ids)) != len(event_ids):
            raise RuntimeError(f'{path}: eventId is not unique')
        norms = np.linalg.norm(directions, axis=1)
        valid = np.isfinite(stops).all(axis=1) & np.isfinite(directions).all(axis=1) & (norms > 1e-6)
        normalized_directions = np.zeros_like(directions, dtype=np.float32)
        normalized_directions[valid] = directions[valid] / norms[valid, None]
        self.cache[key] = {
            int(event_id): (stops[index], normalized_directions[index], bool(valid[index]))
            for index, event_id in enumerate(event_ids)
        }

    def get(self, particle: str, random_seed: int, event_id: int):
        self._load(particle, random_seed)
        try:
            return self.cache[(particle, random_seed)][event_id]
        except KeyError as error:
            raise KeyError(
                f'event_id={event_id} absent from {particle} seed={random_seed}') from error


def graph_identity(graph) -> tuple[str, int, int]:
    required = ('event_id', 'random_seed', 'y')
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise RuntimeError(
            'soft-object truth attachment requires cache provenance; '
            f'missing {missing}')
    particle = 'antiD' if scalar(graph.y) == 1 else 'antiP'
    return particle, scalar(graph.random_seed), scalar(graph.event_id)


def attach_target(graph, lookup: TruthLookup, stop_mean, stop_std) -> bool:
    particle, random_seed, event_id = graph_identity(graph)
    stop, direction, valid = lookup.get(particle, random_seed, event_id)
    graph.mc_soft_truth_valid = torch.tensor([valid], dtype=torch.bool)
    graph.mc_soft_stop_z = torch.tensor(
        (stop - stop_mean) / stop_std if valid else np.zeros(3),
        dtype=torch.float32)
    graph.mc_soft_direction = torch.tensor(
        direction if valid else np.zeros(3), dtype=torch.float32)
    return valid


def fit_stop_normalizer(cache_dir: Path, lookup: TruthLookup,
                        max_graphs_per_shard: int | None):
    total = 0
    sum_stop = np.zeros(3, dtype=np.float64)
    sum_stop_sq = np.zeros(3, dtype=np.float64)
    invalid = 0
    for path in input_shards(cache_dir, 'train'):
        graphs = torch.load(path, map_location='cpu', weights_only=False)
        if max_graphs_per_shard is not None:
            graphs = graphs[:max_graphs_per_shard]
        for graph in graphs:
            particle, random_seed, event_id = graph_identity(graph)
            stop, _, valid = lookup.get(particle, random_seed, event_id)
            if not valid:
                invalid += 1
                continue
            sum_stop += stop
            sum_stop_sq += np.square(stop)
            total += 1
        del graphs
        gc.collect()
    if total == 0:
        raise RuntimeError('no valid stop targets in selected train cache')
    mean = sum_stop / total
    variance = np.maximum(sum_stop_sq / total - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), total, invalid


def write_split(source_dir: Path, output_dir: Path, split: str,
                lookup: TruthLookup, stop_mean: np.ndarray,
                stop_std: np.ndarray, max_graphs_per_shard: int | None):
    total = 0
    valid = 0
    labels: Counter[int] = Counter()
    for source_path in input_shards(source_dir, split):
        graphs = torch.load(source_path, map_location='cpu', weights_only=False)
        if max_graphs_per_shard is not None:
            graphs = graphs[:max_graphs_per_shard]
        for graph in graphs:
            valid += int(attach_target(graph, lookup, stop_mean, stop_std))
            labels[scalar(graph.y)] += 1
        destination = output_dir / source_path.name
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(graphs, temporary)
        temporary.replace(destination)
        with destination.with_suffix('.json').open('w', encoding='utf-8') as handle:
            json.dump({
                'n_graphs': len(graphs),
                'source_shard': str(source_path.resolve()),
                'training_only_fields': [
                    'mc_soft_truth_valid', 'mc_soft_stop_z',
                    'mc_soft_direction'],
            }, handle, indent=2)
        total += len(graphs)
        print(
            f'[{split}] saved {destination.name}: {len(graphs):,} events',
            flush=True)
        del graphs
        gc.collect()
    return {
        'n_graphs': total,
        'valid_targets': valid,
        'valid_fraction': valid / max(total, 1),
        'label_counts': dict(sorted(labels.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-cache-dir', type=Path, required=True)
    parser.add_argument('--root-dir', type=Path,
                        default=Path('/mnt/aohba/GAPS_Sim_2tof'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-graphs-per-shard', type=int,
                        help='smoke test only; omit for the exact full cache')
    args = parser.parse_args()
    if args.max_graphs_per_shard is not None and args.max_graphs_per_shard < 1:
        raise ValueError('--max-graphs-per-shard must be positive')
    if args.output_dir.exists():
        raise FileExistsError(f'refusing to overwrite existing {args.output_dir}')
    args.output_dir.mkdir(parents=True)

    lookup = TruthLookup(args.root_dir)
    stop_mean, stop_std, train_valid, train_invalid = fit_stop_normalizer(
        args.source_cache_dir, lookup, args.max_graphs_per_shard)
    print('soft-object train target statistics:', flush=True)
    print(f'  stop mean: {stop_mean.tolist()}', flush=True)
    print(f'  stop std : {stop_std.tolist()}', flush=True)
    print(f'  valid/invalid: {train_valid:,}/{train_invalid:,}', flush=True)

    manifest = {
        'source_cache_dir': str(args.source_cache_dir.resolve()),
        'root_dir': str(args.root_dir.resolve()),
        'max_graphs_per_shard': args.max_graphs_per_shard,
        'inference_inputs': 'TreeRec graph only; MC fields are training targets only',
        'stop_normalizer': {'mean': stop_mean.tolist(), 'std': stop_std.tolist()},
        'splits': {},
    }
    for split in SPLITS:
        manifest['splits'][split] = write_split(
            args.source_cache_dir, args.output_dir, split, lookup,
            stop_mean, stop_std, args.max_graphs_per_shard)
    (args.output_dir / 'soft_object_truth_cache_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'complete: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()

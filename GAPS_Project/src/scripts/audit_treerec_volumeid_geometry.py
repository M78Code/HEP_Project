"""Check whether fine TreeRec volume IDs add geometry beyond hit positions.

The current graph cache retains only a 22-category coarse volume layer.  This
tool tests the omitted readout-channel part of ``volume_id_`` against binned
hit positions before a categorical channel embedding is considered.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


VOLUME_BRANCH = 'Rec/hitseries_/hitseries_.volume_id_'
POSITION_BRANCH = 'Rec/hitseries_/hitseries_.hit_position_'


def load_hits(path: Path, max_events: int) -> tuple[np.ndarray, np.ndarray]:
    with uproot.open(path) as root_file:
        tree = root_file['TreeRec']
        n_events = min(max_events, tree.num_entries)
        volume_events = tree[VOLUME_BRANCH].array(
            entry_stop=n_events, library='np')
        position_events = tree[POSITION_BRANCH].array(
            entry_stop=n_events, library='ak')

    volumes, positions = [], []
    for volume_event, position_event in zip(volume_events, position_events):
        volume_event = np.asarray(volume_event, dtype=np.int64)
        if not len(volume_event):
            continue
        xyz = np.column_stack([
            np.asarray(position_event['fX'], dtype=np.float32),
            np.asarray(position_event['fY'], dtype=np.float32),
            np.asarray(position_event['fZ'], dtype=np.float32),
        ])
        if len(xyz) != len(volume_event):
            raise RuntimeError(f'{path}: position/volume length mismatch')
        volumes.append(volume_event)
        positions.append(xyz)

    if not volumes:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float32)
    return np.concatenate(volumes), np.concatenate(positions)


def print_category_summary(name: str, values: np.ndarray) -> None:
    counts = Counter(values.tolist())
    top = ', '.join(f'{key}:{count}' for key, count in counts.most_common(12))
    print(f'{name}: categories={len(counts):,}; top={top}')


def coordinate_redundancy(
        category: np.ndarray, positions: np.ndarray, bin_width_mm: float,
        name: str) -> None:
    coordinate_bins = np.rint(positions / bin_width_mm).astype(np.int32)
    members: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    hit_counts: Counter[tuple[int, int, int]] = Counter()
    for key, value in zip(map(tuple, coordinate_bins), category.tolist()):
        members[key].add(int(value))
        hit_counts[key] += 1

    widths = np.fromiter((len(values) for values in members.values()), dtype=np.int64)
    unambiguous_bins = widths == 1
    unambiguous_hits = sum(
        hit_counts[key] for key, values in members.items() if len(values) == 1)
    print(
        f'  {name} at {bin_width_mm:g} mm bins: '
        f'unambiguous bins={int(unambiguous_bins.sum()):,}/{len(members):,} '
        f'({unambiguous_bins.mean():.2%}), '
        f'hits={unambiguous_hits:,}/{len(category):,} '
        f'({unambiguous_hits / len(category):.2%}), '
        f'max categories/bin={widths.max(initial=0)}')


def audit_file(path: Path, max_events: int, bin_widths: list[float]) -> None:
    volume_id, positions = load_hits(path, max_events)
    if not len(volume_id):
        print(f'\n===== {path} =====\nno hits')
        return

    coarse = volume_id // 1_000_000
    channel = volume_id // 1_000
    segment = (volume_id // 1_000) % 1_000
    subchannel = volume_id % 1_000

    print(f'\n===== {path} =====')
    print(f'hits: {len(volume_id):,}')
    print_category_summary('coarse layer (already encoded)', coarse)
    print_category_summary('channel = volume_id // 1000 (candidate)', channel)
    print_category_summary('segment (candidate component)', segment)
    print_category_summary('subchannel (candidate component)', subchannel)

    per_coarse = defaultdict(set)
    for coarse_value, channel_value in zip(coarse.tolist(), channel.tolist()):
        per_coarse[int(coarse_value)].add(int(channel_value))
    print('channels per coarse layer:')
    print('  ' + ', '.join(
        f'{layer}:{len(channels)}' for layer, channels in sorted(per_coarse.items())))

    print('position-bin ambiguity (one category per bin means position already identifies it):')
    for width in bin_widths:
        coordinate_redundancy(channel, positions, width, 'channel')
        coordinate_redundancy(subchannel, positions, width, 'subchannel')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--max-events', type=int, default=10_000)
    parser.add_argument('--position-bins-mm', type=float, nargs='+',
                        default=[1.0, 10.0, 25.0])
    args = parser.parse_args()
    if args.max_events < 1:
        raise ValueError('--max-events must be positive')
    if any(width <= 0.0 for width in args.position_bins_mm):
        raise ValueError('--position-bins-mm must be positive')
    for path in args.input:
        audit_file(path, args.max_events, args.position_bins_mm)


if __name__ == '__main__':
    main()

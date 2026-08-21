"""Audit TreeRec hitseries metadata that is not used by the current GNN input.

The current TreeRec cache consumes position, energy deposition, hit time, and a
coarse detector/layer value derived from ``volume_id_``.  This read-only tool
checks whether the remaining hitseries branches carry usable reconstructed
information before they are considered as model features.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


HITSERIES = 'Rec/hitseries_/'
BRANCHES = {
    'volume_id': HITSERIES + 'hitseries_.volume_id_',
    'index': HITSERIES + 'hitseries_.index_',
    'f_unique_id': HITSERIES + 'hitseries_.fUniqueID',
    'f_bits': HITSERIES + 'hitseries_.fBits',
}


def flattened(array: ak.Array) -> np.ndarray:
    """Return a one-dimensional numeric view of an event-jagged branch."""
    return ak.to_numpy(ak.flatten(array, axis=None))


def numeric_summary(values: np.ndarray) -> str:
    if len(values) == 0:
        return 'no values'
    values = np.asarray(values)
    return (
        f'n={len(values):,} unique={len(np.unique(values)):,} '
        f'min={values.min()} p50={np.median(values):.6g} max={values.max()} '
        f'nonzero={int(np.count_nonzero(values)):,}'
    )


def print_top_values(name: str, values: np.ndarray, limit: int = 12) -> None:
    counts = Counter(np.asarray(values).tolist())
    top = ', '.join(
        f'{value}:{count}' for value, count in counts.most_common(limit))
    print(f'  {name} top values: {top or "none"}')


def audit_file(path: Path, max_events: int) -> None:
    with uproot.open(path) as root_file:
        tree = root_file['TreeRec']
        missing = [branch for branch in BRANCHES.values() if branch not in tree]
        if missing:
            raise KeyError(f'{path}: missing branches: {missing}')

        n_events = min(max_events, tree.num_entries)
        arrays = tree.arrays(
            list(BRANCHES.values()), entry_stop=n_events, library='ak')

    print(f'\n===== {path} =====')
    print(f'events checked: {n_events:,}')

    counts = {
        name: ak.to_numpy(ak.num(arrays[branch], axis=1))
        for name, branch in BRANCHES.items()
    }
    hit_counts = counts['volume_id']
    print(
        'hits/event min/median/max: '
        f'{hit_counts.min(initial=0)} / {np.median(hit_counts):.1f} / '
        f'{hit_counts.max(initial=0)}')
    for name, value in counts.items():
        mismatch = int(np.count_nonzero(value != hit_counts))
        print(f'length matches volume_id ({name}): {n_events - mismatch:,}/{n_events:,}')

    values = {
        name: flattened(arrays[branch])
        for name, branch in BRANCHES.items()
    }
    for name in ('index', 'f_unique_id', 'f_bits'):
        print(f'{name}: {numeric_summary(values[name])}')
        print_top_values(name, values[name])

    index_is_local_permutation = 0
    for event_index, event_ids in enumerate(arrays[BRANCHES['index']]):
        event_ids = np.asarray(event_ids, dtype=np.int64)
        if len(event_ids) and np.array_equal(
                np.sort(event_ids), np.arange(len(event_ids), dtype=np.int64)):
            index_is_local_permutation += 1
    nonempty_events = int(np.count_nonzero(hit_counts))
    print(
        'index is a 0..N-1 permutation: '
        f'{index_is_local_permutation:,}/{nonempty_events:,} nonempty events')

    volume_id = values['volume_id'].astype(np.int64, copy=False)
    coarse = volume_id // 1_000_000
    segment = (volume_id // 1_000) % 1_000
    subchannel = volume_id % 1_000
    print('volume_id components:')
    for name, component in (
            ('coarse (currently encoded)', coarse),
            ('segment (currently omitted)', segment),
            ('subchannel (currently omitted)', subchannel)):
        print(f'  {name}: {numeric_summary(component)}')
        print_top_values(name, component)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=Path, required=True, nargs='+',
        help='one or more TreeRec ROOT files')
    parser.add_argument('--max-events', type=int, default=10_000)
    args = parser.parse_args()
    if args.max_events < 1:
        raise ValueError('--max-events must be positive')
    for path in args.input:
        audit_file(path, args.max_events)


if __name__ == '__main__':
    main()

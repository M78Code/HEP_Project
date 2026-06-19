"""Audit reconstructed primary-particle fields in the Aohba ROOT dataset.

The script reads a small sample only. MC fields are used for diagnostics and
are never proposed as model inputs.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import uproot


DEFAULT_ROOT_DIR = Path('/mnt/aohba/GAPS_Sim_2tof')

BRANCHES = {
    'mc_beta': 'Mc/CEventBase/primaryBetaGenerated_',
    'mc_stopping_volume': 'Mc/primaryStoppingVolume_',
    'event_id': 'Rec/CEventBase/eventId_',
    'active_reco': 'Rec/activeReco_',
    'event_quality': 'Rec/event_quality',
    'rec_beta': 'Rec/primaryBeta_/primaryBeta_.second',
    'rec_stopping_volume': (
        'Rec/primaryStoppingVolume_/primaryStoppingVolume_.second'
    ),
    'rec_direction': (
        'Rec/primaryMomentumDirection_/primaryMomentumDirection_.second'
    ),
    'rec_primary_edep': (
        'Rec/primaryEnergyDepositions_/primaryEnergyDepositions_.second'
    ),
    'rec_hit_track_index': 'Rec/HitTrackIndex/HitTrackIndex.second',
    'hit_volume_id': 'Rec/hitseries_/hitseries_.volume_id_',
}


def to_numeric_array(value, dtype=np.float64):
    try:
        return np.asarray(value, dtype=dtype).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=dtype)


def finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {'n': 0}
    return {
        'n': int(len(finite)),
        'min': float(finite.min()),
        'p01': float(np.quantile(finite, 0.01)),
        'median': float(np.median(finite)),
        'mean': float(finite.mean()),
        'p99': float(np.quantile(finite, 0.99)),
        'max': float(finite.max()),
    }


def multiplicity_summary(counts):
    counts = np.asarray(counts, dtype=np.int64)
    total = max(1, len(counts))
    distribution = Counter(counts.tolist())
    return {
        'events': int(len(counts)),
        'zero': int((counts == 0).sum()),
        'one': int((counts == 1).sum()),
        'multiple': int((counts > 1).sum()),
        'available_fraction': float((counts > 0).sum() / total),
        'single_fraction': float((counts == 1).sum() / total),
        'max_candidates': int(counts.max(initial=0)),
        'count_distribution_first10': {
            str(key): int(distribution[key])
            for key in sorted(distribution)[:10]
        },
    }


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def first_candidate_numeric(array):
    values = []
    counts = []
    for event_value in array:
        numeric = to_numeric_array(event_value)
        counts.append(len(numeric))
        values.append(float(numeric[0]) if len(numeric) else np.nan)
    return np.asarray(values), np.asarray(counts)


def direction_summary(array):
    counts = []
    norms = []
    xyz = []
    for event_value in array:
        try:
            count = len(event_value)
        except TypeError:
            count = 0
        counts.append(count)
        if count == 0:
            continue
        candidate = event_value[0]
        try:
            vector = np.asarray([
                float(candidate['fX']),
                float(candidate['fY']),
                float(candidate['fZ']),
            ])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if np.isfinite(vector).all():
            xyz.append(vector)
            norms.append(float(np.linalg.norm(vector)))

    result = {
        'multiplicity': multiplicity_summary(counts),
        'norm': finite_summary(norms),
    }
    if xyz:
        xyz = np.asarray(xyz)
        result['component_mean'] = xyz.mean(axis=0).tolist()
        result['component_std'] = xyz.std(axis=0).tolist()
    return result


def nested_numeric_summary(array):
    event_counts = []
    scalar_counts = []
    sums = []
    for event_value in array:
        try:
            event_counts.append(len(event_value))
        except TypeError:
            event_counts.append(0)

        scalars = []
        try:
            for candidate in event_value:
                numeric = to_numeric_array(candidate)
                scalars.extend(numeric[np.isfinite(numeric)].tolist())
        except TypeError:
            pass
        scalar_counts.append(len(scalars))
        sums.append(float(np.sum(scalars)) if scalars else np.nan)

    return {
        'candidate_multiplicity': multiplicity_summary(event_counts),
        'scalar_count_per_event': finite_summary(scalar_counts),
        'sum_per_event': finite_summary(sums),
    }


def audit_file(path, max_events):
    with uproot.open(path) as root_file:
        mc = root_file['TreeMc']
        rec = root_file['TreeRec']
        n_events = min(max_events, mc.num_entries, rec.num_entries)

        missing = []
        for name, branch in BRANCHES.items():
            tree = mc if name.startswith('mc_') else rec
            if branch not in tree:
                missing.append(branch)
        if missing:
            raise KeyError(f'missing branches in {path.name}: {missing}')

        arrays = {
            'mc_beta': mc[BRANCHES['mc_beta']].array(entry_stop=n_events),
            'mc_stopping_volume': mc[
                BRANCHES['mc_stopping_volume']
            ].array(entry_stop=n_events),
        }
        for name in BRANCHES:
            if name.startswith('mc_'):
                continue
            arrays[name] = rec[BRANCHES[name]].array(entry_stop=n_events)

    mc_beta = np.asarray(arrays['mc_beta'], dtype=np.float64)
    mc_stop = np.asarray(arrays['mc_stopping_volume'], dtype=np.int64)
    rec_beta, rec_beta_counts = first_candidate_numeric(arrays['rec_beta'])
    rec_stop, rec_stop_counts = first_candidate_numeric(
        arrays['rec_stopping_volume'])

    beta_mask = np.isfinite(rec_beta)
    stop_mask = np.isfinite(rec_stop)
    rec_stop_int = np.zeros(n_events, dtype=np.int64)
    rec_stop_int[stop_mask] = rec_stop[stop_mask].astype(np.int64)

    mc_tracker = (mc_stop // 1_000_000) >= 200
    rec_tracker = (rec_stop_int // 1_000_000) >= 200
    stop_system_match = (
        (mc_stop // 100_000_000)
        == (rec_stop_int // 100_000_000)
    )

    hit_counts = np.asarray(
        [len(value) for value in arrays['hit_volume_id']], dtype=np.int64)
    track_index_counts = np.asarray(
        [len(value) for value in arrays['rec_hit_track_index']], dtype=np.int64)
    track_index_values = np.concatenate([
        to_numeric_array(value, dtype=np.int64)
        for value in arrays['rec_hit_track_index']
        if len(value) > 0
    ]) if np.any(track_index_counts > 0) else np.asarray([], dtype=np.int64)

    event_quality_counts = np.asarray(
        [len(value) for value in arrays['event_quality']], dtype=np.int64)
    event_quality_values = np.concatenate([
        to_numeric_array(value, dtype=np.int64)
        for value in arrays['event_quality']
        if len(value) > 0
    ]) if np.any(event_quality_counts > 0) else np.asarray([], dtype=np.int64)

    active_reco_values = [str(value) for value in arrays['active_reco']]
    active_reco_counts = Counter(active_reco_values)

    result = {
        'file': str(path),
        'events': int(n_events),
        'mc_beta': finite_summary(mc_beta),
        'rec_beta': {
            'multiplicity': multiplicity_summary(rec_beta_counts),
            'first_candidate': finite_summary(rec_beta[beta_mask]),
            'correlation_with_mc_first': correlation(
                rec_beta[beta_mask], mc_beta[beta_mask]),
            'difference_first_minus_mc': finite_summary(
                rec_beta[beta_mask] - mc_beta[beta_mask]),
        },
        'rec_stopping_volume': {
            'multiplicity': multiplicity_summary(rec_stop_counts),
            'exact_match_fraction_first': (
                float((rec_stop_int[stop_mask] == mc_stop[stop_mask]).mean())
                if stop_mask.any() else None
            ),
            'detector_system_match_fraction_first': (
                float(stop_system_match[stop_mask].mean())
                if stop_mask.any() else None
            ),
            'mc_tracker_atrest_fraction': float(mc_tracker.mean()),
            'rec_tracker_atrest_fraction_first': (
                float(rec_tracker[stop_mask].mean())
                if stop_mask.any() else None
            ),
        },
        'rec_direction': direction_summary(arrays['rec_direction']),
        'rec_primary_edep': nested_numeric_summary(
            arrays['rec_primary_edep']),
        'hit_track_index': {
            'multiplicity': multiplicity_summary(track_index_counts),
            'hit_count': finite_summary(hit_counts),
            'same_length_as_hitseries_fraction': float(
                (track_index_counts == hit_counts).mean()),
            'values': finite_summary(track_index_values),
            'negative_fraction': (
                float((track_index_values < 0).mean())
                if len(track_index_values) else None
            ),
        },
        'event_quality': {
            'multiplicity': multiplicity_summary(event_quality_counts),
            'values': finite_summary(event_quality_values),
            'value_counts_first20': {
                str(key): int(value)
                for key, value in sorted(
                    Counter(event_quality_values.tolist()).items()
                )[:20]
            },
        },
        'active_reco_top10': dict(active_reco_counts.most_common(10)),
    }
    return result


def print_summary(particle, result):
    print(f'\n{"=" * 78}')
    print(f'{particle}: {Path(result["file"]).name}')
    print(f'events: {result["events"]:,}')
    print(f'{"=" * 78}')

    beta = result['rec_beta']
    print('\n[Rec primary beta]')
    print(json.dumps(beta, indent=2, ensure_ascii=False))

    stopping = result['rec_stopping_volume']
    print('\n[Rec stopping volume]')
    print(json.dumps(stopping, indent=2, ensure_ascii=False))

    print('\n[Rec primary direction]')
    print(json.dumps(result['rec_direction'], indent=2, ensure_ascii=False))

    print('\n[Rec primary energy depositions]')
    print(json.dumps(result['rec_primary_edep'], indent=2, ensure_ascii=False))

    print('\n[HitTrackIndex]')
    print(json.dumps(result['hit_track_index'], indent=2, ensure_ascii=False))

    print('\n[event_quality / activeReco]')
    print(json.dumps({
        'event_quality': result['event_quality'],
        'active_reco_top10': result['active_reco_top10'],
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=Path, default=DEFAULT_ROOT_DIR)
    parser.add_argument('--events', type=int, default=10_000)
    parser.add_argument('--file-index', type=int, default=0)
    parser.add_argument(
        '--particles', nargs='+', default=['antiD', 'antiP'],
        choices=['antiD', 'antiP'],
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path.home() / 'aohba_rec_primary_audit.json',
    )
    args = parser.parse_args()

    results = {}
    for particle in args.particles:
        files = sorted((args.root_dir / particle).glob('*.root'))
        if args.file_index >= len(files):
            raise IndexError(
                f'{particle}: file index {args.file_index} out of '
                f'range for {len(files)} files')
        result = audit_file(files[args.file_index], args.events)
        results[particle] = result
        print_summary(particle, result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f'\nJSON saved: {args.output}')


if __name__ == '__main__':
    main()

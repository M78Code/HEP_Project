"""Audit TreeMc targets for training-only soft-object supervision.

This is a diagnostic only.  It never writes labels into a graph cache and must
not be used as an inference-time feature.  It checks whether the MC stopping
point and primary direction are well defined and spatially reachable from the
observed TreeRec hit cloud.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


MC_BRANCHES = (
    'Mc/primaryPosition_',
    'Mc/primaryStoppingPosition_',
    'Mc/primaryStoppingTime_',
    'Mc/CEventBase/primaryMomentumDirectionGenerated_',
    'Mc/hitTrackIndex_',
    'Mc/meanPosition_',
)
REC_BRANCHES = (
    'Rec/hitseries_/hitseries_.volume_id_',
    'Rec/hitseries_/hitseries_.energydep_',
    'Rec/hitseries_/hitseries_.hit_position_',
)


def vector_components(values) -> np.ndarray:
    """Convert a scalar or jagged ROOT TVector3 branch to a trailing xyz axis."""
    return ak.concatenate(
        [
            values['fX'][..., np.newaxis],
            values['fY'][..., np.newaxis],
            values['fZ'][..., np.newaxis],
        ],
        axis=-1,
    )


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {'n': 0}
    return {
        'n': int(len(array)),
        'p01': float(np.quantile(array, 0.01)),
        'p10': float(np.quantile(array, 0.10)),
        'median': float(np.median(array)),
        'p90': float(np.quantile(array, 0.90)),
        'p99': float(np.quantile(array, 0.99)),
    }


def audit_file(path: Path, max_events: int) -> dict:
    with uproot.open(path) as root_file:
        mc = root_file['TreeMc']
        rec = root_file['TreeRec']
        events = min(max_events, mc.num_entries, rec.num_entries)
        missing = [
            branch for branch in MC_BRANCHES if branch not in mc
        ] + [branch for branch in REC_BRANCHES if branch not in rec]
        if missing:
            raise KeyError(f'{path}: missing branches: {missing}')

        mc_arrays = mc.arrays(MC_BRANCHES, entry_stop=events, library='ak')
        rec_arrays = rec.arrays(REC_BRANCHES, entry_stop=events, library='ak')

    primary = vector_components(mc_arrays['Mc/primaryPosition_'])
    stopping = vector_components(mc_arrays['Mc/primaryStoppingPosition_'])
    direction = vector_components(
        mc_arrays['Mc/CEventBase/primaryMomentumDirectionGenerated_'])
    hit_position = vector_components(
        rec_arrays['Rec/hitseries_/hitseries_.hit_position_'])
    volume_id = rec_arrays['Rec/hitseries_/hitseries_.volume_id_']
    energy = rec_arrays['Rec/hitseries_/hitseries_.energydep_']
    mc_track_index = mc_arrays['Mc/hitTrackIndex_']
    mc_mean_position = vector_components(mc_arrays['Mc/meanPosition_'])

    nearest_all_mm = []
    nearest_sili_mm = []
    silicentroid_mm = []
    direction_path_cosine = []
    rec_hit_counts = []
    sili_hit_counts = []
    events_with_sili_hits = 0
    mc_track_alignment = []
    valid_stopping = 0
    valid_direction = 0

    for event in range(events):
        stop = np.asarray(stopping[event], dtype=np.float64)
        start = np.asarray(primary[event], dtype=np.float64)
        unit = np.asarray(direction[event], dtype=np.float64)
        positions = np.asarray(hit_position[event], dtype=np.float64)
        volumes = np.asarray(volume_id[event], dtype=np.int64)
        deposits = np.asarray(energy[event], dtype=np.float64)
        rec_hit_counts.append(len(positions))

        mc_indices = np.asarray(mc_track_index[event], dtype=np.int64)
        mc_positions = np.asarray(mc_mean_position[event], dtype=np.float64)
        mc_track_alignment.append(int(len(mc_indices) == len(mc_positions)))

        if not (np.isfinite(stop).all() and np.isfinite(start).all()):
            continue
        valid_stopping += 1
        if len(positions):
            distances = np.linalg.norm(positions - stop, axis=1)
            nearest_all_mm.append(float(distances.min()))

        # In production volume IDs, detector system 2 denotes Si(Li).
        sili = (volumes // 100_000_000) == 2
        sili_positions = positions[sili]
        sili_energy = deposits[sili]
        sili_hit_counts.append(int(len(sili_positions)))
        if len(sili_positions):
            events_with_sili_hits += 1
            sili_distances = np.linalg.norm(sili_positions - stop, axis=1)
            nearest_sili_mm.append(float(sili_distances.min()))
            weights = np.maximum(sili_energy, 0.0)
            if np.isfinite(weights).all() and weights.sum() > 0:
                centroid = np.average(sili_positions, axis=0, weights=weights)
                silicentroid_mm.append(float(np.linalg.norm(centroid - stop)))

        norm = float(np.linalg.norm(unit))
        path = stop - start
        path_norm = float(np.linalg.norm(path))
        if np.isfinite(unit).all() and norm > 1e-9 and path_norm > 1e-9:
            valid_direction += 1
            direction_path_cosine.append(float(np.dot(unit, path) / (norm * path_norm)))

    return {
        'file': str(path),
        'events_checked': events,
        'tree_entry_counts': {'TreeMc': int(mc.num_entries), 'TreeRec': int(rec.num_entries)},
        'stopping_target': {
            'finite_fraction': valid_stopping / max(events, 1),
            'nearest_any_treerec_hit_mm': summary(nearest_all_mm),
            'nearest_sili_hit_mm': summary(nearest_sili_mm),
            'energy_weighted_sili_centroid_distance_mm': summary(silicentroid_mm),
            'events_with_sili_hits_fraction': events_with_sili_hits / max(valid_stopping, 1),
        },
        'direction_target': {
            'finite_fraction': valid_direction / max(events, 1),
            'mc_direction_vs_primary_to_stop_cosine': summary(direction_path_cosine),
        },
        'multiplicities': {
            'treerec_hit_count': summary(rec_hit_counts),
            'sili_hit_count': summary(sili_hit_counts),
            'mc_hitTrackIndex_length_matches_meanPosition_fraction': (
                float(np.mean(mc_track_alignment)) if mc_track_alignment else None),
        },
        'interpretation': (
            'Stopping position and primary direction may be used only as '
            'training-time auxiliary targets. TreeRec-to-MC per-hit matching '
            'is not established by this audit.'
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--max-events', type=int, default=10_000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    report = {}
    for path in args.input:
        result = audit_file(path, args.max_events)
        report[str(path)] = result
        print(f'\n===== {path} =====')
        print(json.dumps(result, indent=2), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'\nsaved: {args.output}')


if __name__ == '__main__':
    main()

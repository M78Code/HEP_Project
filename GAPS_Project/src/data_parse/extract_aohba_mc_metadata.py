"""Extract compact MC metadata sidecars aligned with preprocessed Aohba PKLs.

The sidecars are for dataset diagnostics and event selection only. Their
contents must not be passed to the classifier as input features.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import uproot


DEFAULT_ROOT_DIR = Path('/mnt/aohba/GAPS_Sim_2tof')
DEFAULT_PKL_DIR = Path('/mnt/ynakagami3/aohba_preprocess')
DEFAULT_OUTPUT_DIR = Path('/mnt/ynakagami3/aohba_preprocess/mc_metadata')


def load_expected_events(summary_path: Path) -> int:
    if not summary_path.exists():
        raise FileNotFoundError(f'missing summary: {summary_path}')
    with open(summary_path, encoding='utf-8') as file:
        return int(json.load(file)['total_events'])


def extract_one(root_path: Path, summary_path: Path, output_path: Path):
    expected_events = load_expected_events(summary_path)
    t0 = time.time()

    with uproot.open(root_path) as root_file:
        mc = root_file['TreeMc']
        stopping_vol = np.asarray(
            mc['Mc/primaryStoppingVolume_'].array(), dtype=np.int32)
        event_id = np.asarray(
            mc['Mc/CEventBase/eventId_'].array(), dtype=np.uint32)
        random_seed = np.asarray(
            mc['Mc/randomSeed_'].array(), dtype=np.uint32)

    lengths = {
        len(stopping_vol),
        len(event_id),
        len(random_seed),
        expected_events,
    }
    if len(lengths) != 1:
        raise RuntimeError(
            f'{root_path.name}: event count mismatch; '
            f'ROOT={len(stopping_vol)}, summary={expected_events}')

    detector_system = stopping_vol.astype(np.int64) // 100_000_000
    system_values, system_counts = np.unique(
        detector_system, return_counts=True)
    system_distribution = {
        str(int(key)): int(value)
        for key, value in zip(system_values, system_counts)
    }
    is_tracker_atrest = detector_system == 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix('.tmp')
    with open(temporary_path, 'wb') as file:
        np.savez(
            file,
            stopping_vol=stopping_vol,
            is_tracker_atrest=is_tracker_atrest,
            event_id=event_id,
            random_seed=random_seed,
        )
    temporary_path.replace(output_path)

    summary = {
        'source_root': root_path.name,
        'source_summary': summary_path.name,
        'events': expected_events,
        'tracker_atrest_events': int(is_tracker_atrest.sum()),
        'tracker_atrest_fraction': float(is_tracker_atrest.mean()),
        'stopping_system_distribution': system_distribution,
        'size_mb': round(output_path.stat().st_size / 1024 / 1024, 2),
        'elapsed_sec': round(time.time() - t0, 2),
    }
    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=Path, default=DEFAULT_ROOT_DIR)
    parser.add_argument('--pkl-dir', type=Path, default=DEFAULT_PKL_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--particles', nargs='+', default=['antiD', 'antiP'],
        choices=['antiD', 'antiP'],
    )
    parser.add_argument('--max-files', type=int)
    parser.add_argument(
        '--overwrite', action='store_true',
        help='replace existing sidecars',
    )
    args = parser.parse_args()

    overall = []
    for particle in args.particles:
        pkl_files = sorted((args.pkl_dir / particle).glob('*.pkl'))
        if args.max_files is not None:
            pkl_files = pkl_files[:args.max_files]
        if not pkl_files:
            raise FileNotFoundError(
                f'no PKL files under {args.pkl_dir / particle}')

        print(f'\n=== {particle}: {len(pkl_files)} files ===', flush=True)
        for index, pkl_path in enumerate(pkl_files, 1):
            root_path = args.root_dir / particle / f'{pkl_path.stem}.root'
            summary_path = (
                args.pkl_dir / particle / f'{pkl_path.stem}_summary.json'
            )
            output_path = (
                args.output_dir / particle / f'{pkl_path.stem}.npz'
            )

            if output_path.exists() and not args.overwrite:
                print(
                    f'[{index:03d}/{len(pkl_files):03d}] '
                    f'skip existing: {output_path.name}',
                    flush=True,
                )
                continue
            if not root_path.exists():
                raise FileNotFoundError(f'missing ROOT: {root_path}')

            row = extract_one(root_path, summary_path, output_path)
            row['particle'] = particle
            overall.append(row)
            print(
                f'[{index:03d}/{len(pkl_files):03d}] {pkl_path.stem}: '
                f'{row["events"]:,} events, '
                f'at-rest={row["tracker_atrest_fraction"]:.2%}, '
                f'{row["size_mb"]:.2f} MB, '
                f'{row["elapsed_sec"]:.1f}s',
                flush=True,
            )

    if overall:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(
            args.output_dir / 'extraction_summary.json',
            'w',
            encoding='utf-8',
        ) as file:
            json.dump(overall, file, indent=2)

        total_events = sum(row['events'] for row in overall)
        total_atrest = sum(row['tracker_atrest_events'] for row in overall)
        print('\n========== complete ==========')
        print(f'files: {len(overall)}')
        print(f'events: {total_events:,}')
        print(f'tracker at-rest: {total_atrest:,} '
              f'({total_atrest / total_events:.2%})')
        print(f'output: {args.output_dir}')


if __name__ == '__main__':
    main()

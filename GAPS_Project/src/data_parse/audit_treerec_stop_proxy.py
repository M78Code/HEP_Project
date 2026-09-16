#!/usr/bin/env python3
"""Audit TreeRec-only candidates for a reconstructible stopping proxy.

The input directories are provenance exports whose source ROOT entries are
already fixed.  This script deliberately reads TreeRec only: it never opens
TreeMc and therefore cannot use truth stopping volume, kinetic energy, or
zero-step information as a proxy feature.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


PARTICLES = ("antiP", "antiD")
REC_BRANCHES = {
    "stop_volume": "Rec/primaryStoppingVolume_/primaryStoppingVolume_.second",
    "stop_time": "Rec/primaryStoppingTime_/primaryStoppingTime_.second",
    "beta": "Rec/primaryBeta_/primaryBeta_.second",
    "hit_volume": "Rec/hitseries_/hitseries_.volume_id_",
    "hit_energy": "Rec/hitseries_/hitseries_.energydep_",
    "hit_time": "Rec/hitseries_/hitseries_.hit_time_",
}


def parse_group(value: str) -> tuple[str, Path]:
    name, separator, directory = value.partition("=")
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError(
            "--group must have the form NAME=PROVENANCE_DIRECTORY"
        )
    return name, Path(directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=parse_group, action="append", required=True)
    parser.add_argument("--events-per-class", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def first_finite(values) -> float:
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric[0]) if len(numeric) else np.nan


def summary(values: list[float]) -> dict[str, float | int | None]:
    numeric = np.asarray(values, dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) == 0:
        return {"n": 0, "p05": None, "median": None, "p95": None}
    return {
        "n": int(len(numeric)),
        "p05": float(np.quantile(numeric, 0.05)),
        "median": float(np.median(numeric)),
        "p95": float(np.quantile(numeric, 0.95)),
    }


def load_provenance(directory: Path, particle: str, limit: int) -> dict:
    source = directory / particle
    if not (source / "_SUCCESS").is_file():
        raise RuntimeError(f"incomplete provenance: {source}")
    file_indices = np.load(source / "source_file_indices.npy", mmap_mode="r")
    entries = np.load(source / "source_entries.npy", mmap_mode="r")
    files = [
        Path(line) for line in (source / "source_files.txt").read_text().splitlines()
        if line.strip()
    ]
    count = min(limit, len(entries))
    if count == 0:
        raise RuntimeError(f"no events in {source}")
    return {
        "file_indices": np.asarray(file_indices[:count], dtype=np.int64),
        "entries": np.asarray(entries[:count], dtype=np.int64),
        "files": files,
    }


def collect_group_particle(provenance: dict) -> dict:
    file_indices = provenance["file_indices"]
    entries = provenance["entries"]
    records: list[dict | None] = [None] * len(entries)

    for file_index in np.unique(file_indices):
        output_indices = np.flatnonzero(file_indices == file_index)
        root_entries = entries[output_indices]
        entry_start = int(root_entries.min())
        entry_stop = int(root_entries.max()) + 1
        root_path = provenance["files"][int(file_index)]

        with uproot.open(root_path) as root_file:
            rec = root_file["TreeRec"]
            arrays = {
                name: rec[branch].array(
                    entry_start=entry_start, entry_stop=entry_stop, library="ak"
                )
                for name, branch in REC_BRANCHES.items()
            }

        for output_index, root_entry in zip(output_indices, root_entries):
            local = int(root_entry) - entry_start
            stop_volumes = np.asarray(arrays["stop_volume"][local], dtype=np.int64)
            stop_times = np.asarray(arrays["stop_time"][local], dtype=np.float64)
            beta = np.asarray(arrays["beta"][local], dtype=np.float64)
            volume = np.asarray(arrays["hit_volume"][local], dtype=np.int64)
            energy = np.asarray(arrays["hit_energy"][local], dtype=np.float64)
            hit_time = np.asarray(arrays["hit_time"][local], dtype=np.float64)
            if not (len(volume) == len(energy) == len(hit_time)):
                raise RuntimeError(f"{root_path}: hit-array mismatch at {root_entry}")

            tracker = (volume // 100_000_000) == 2
            tracker_times = hit_time[tracker & np.isfinite(hit_time)]
            records[int(output_index)] = {
                "rec_stop_available": bool(len(stop_volumes)),
                "rec_stop_tracker_first": bool(
                    len(stop_volumes) and stop_volumes[0] // 100_000_000 == 2
                ),
                "rec_stop_tracker_any": bool(
                    np.any(stop_volumes // 100_000_000 == 2)
                ),
                "rec_stop_time_available": bool(np.isfinite(stop_times).any()),
                "rec_beta_available": bool(np.isfinite(beta).any()),
                "n_hits": float(len(volume)),
                "n_tracker_hits": float(tracker.sum()),
                "tracker_energy": float(energy[tracker].sum()),
                "last_tracker_hit_time": (
                    float(tracker_times.max()) if len(tracker_times) else np.nan
                ),
            }

    if any(record is None for record in records):
        raise RuntimeError("failed to read every selected TreeRec event")
    return summarise_records(records)


def summarise_records(records: list[dict]) -> dict:
    total = len(records)
    flags = (
        "rec_stop_available",
        "rec_stop_tracker_first",
        "rec_stop_tracker_any",
        "rec_stop_time_available",
        "rec_beta_available",
    )
    result = {
        "events": total,
        "fractions": {
            name: float(sum(record[name] for record in records) / total)
            for name in flags
        },
        "continuous": {
            name: summary([record[name] for record in records])
            for name in (
                "n_hits",
                "n_tracker_hits",
                "tracker_energy",
                "last_tracker_hit_time",
            )
        },
    }
    return result


def print_result(results: dict[str, dict]) -> None:
    print("\n===== TreeRec-only stopping-proxy audit =====")
    print("No TreeMc stopping, kinetic-energy, or zero-step branch was read.\n")
    for group, particles in results.items():
        print(f"[{group}]")
        for particle, values in particles.items():
            fractions = values["fractions"]
            continuous = values["continuous"]
            print(
                f"  {particle}: N={values['events']:,}  "
                f"Rec stop available={fractions['rec_stop_available']:.2%}  "
                f"Rec stop in tracker(first/any)="
                f"{fractions['rec_stop_tracker_first']:.2%}/"
                f"{fractions['rec_stop_tracker_any']:.2%}"
            )
            print(
                "    median: "
                f"hits={continuous['n_hits']['median']:.1f}, "
                f"tracker hits={continuous['n_tracker_hits']['median']:.1f}, "
                f"tracker Edep={continuous['tracker_energy']['median']:.4g}, "
                f"last tracker time={continuous['last_tracker_hit_time']['median']:.4g}"
            )

    names = list(results)
    if len(names) == 2:
        reference, selected = names
        print(f"\n===== Enrichment: {selected} / {reference} =====")
        for particle in PARTICLES:
            print(f"{particle}:")
            for key in results[reference][particle]["fractions"]:
                base = results[reference][particle]["fractions"][key]
                strict = results[selected][particle]["fractions"][key]
                ratio = strict / base if base else float("inf")
                print(f"  {key:30s} {base:.2%} -> {strict:.2%}  ratio={ratio:.3f}")


def main() -> None:
    args = parse_args()
    if args.events_per_class < 1:
        raise ValueError("--events-per-class must be positive")
    groups = dict(args.group)
    if len(groups) != len(args.group):
        raise ValueError("group names must be unique")

    results: dict[str, dict] = defaultdict(dict)
    for group, directory in groups.items():
        for particle in PARTICLES:
            provenance = load_provenance(directory, particle, args.events_per_class)
            results[group][particle] = collect_group_particle(provenance)

    results = dict(results)
    print_result(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nJSON: {args.output}")
    print("TREEREC STOPPING-PROXY AUDIT: COMPLETE")


if __name__ == "__main__":
    main()

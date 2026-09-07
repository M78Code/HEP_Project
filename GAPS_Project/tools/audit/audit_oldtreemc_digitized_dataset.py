#!/usr/bin/env python3
"""Audit a seeded old-TreeMc digitization-only ROOT dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


REQUIRED_HITSERIES = (
    "Rec/hitseries_/hitseries_.volume_id_",
    "Rec/hitseries_/hitseries_.energydep_",
    "Rec/hitseries_/hitseries_.hit_position_",
    "Rec/hitseries_/hitseries_.hit_time_",
)
VOLUME_BRANCH = REQUIRED_HITSERIES[0]
ENERGY_BRANCH = REQUIRED_HITSERIES[1]
NAME_PATTERN = re.compile(
    r"^reco_(train|val|test)_(antip|antid)_(\d+)_"
    r"selected(\d+)_shard(\d+)\.root$"
)
FALLBACK_PATTERNS = (
    "Non-finite tracker ADC response",
    "Unable to invert tracker ADC response",
)
FATAL_PATTERNS = (
    "segmentation",
    "fatal",
    "terminate called",
    "runtime_error",
    "bad_alloc",
    "OUTPUT VALIDATION FAILED",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--digitized-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument(
        "--expected-split",
        action="append",
        required=True,
        metavar="SPLIT=EVENTS",
    )
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_expected_splits(values: list[str]) -> dict[str, int]:
    expected = {}
    for value in values:
        split, separator, raw_events = value.partition("=")
        if not separator or split not in {"train", "val", "test"}:
            raise ValueError("--expected-split must be train|val|test=EVENTS")
        if split in expected:
            raise ValueError(f"duplicate expected split: {split}")
        expected[split] = int(raw_events)
    if set(expected) != {"train", "val", "test"}:
        raise ValueError("expected split counts are required for train, val, test")
    return expected


def count_log_patterns(log_dir: Path) -> tuple[Counter, Counter]:
    fallback = Counter()
    fatal = Counter()
    for path in sorted(log_dir.glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FALLBACK_PATTERNS:
            fallback[pattern] += text.count(pattern)
        lower = text.lower()
        for pattern in FATAL_PATTERNS:
            fatal[pattern] += lower.count(pattern.lower())
    return fallback, fatal


def main() -> None:
    args = parse_args()
    if args.expected_files < 1 or args.expected_events < 1:
        raise ValueError("expected files/events must be positive")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    expected_splits = parse_expected_splits(args.expected_split)

    manifest_path = args.digitized_dir / "digitization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_seeds = manifest.get("digitization_seeds", {})
    expected_names = {f"reco_{name}" for name in manifest_seeds}
    paths = sorted(args.digitized_dir.glob("reco_*.root"))
    actual_names = {path.name for path in paths}
    issues = []

    if manifest.get("files") != args.expected_files:
        issues.append(f"manifest files={manifest.get('files')}")
    if manifest.get("events") != args.expected_events:
        issues.append(f"manifest events={manifest.get('events')}")
    if manifest.get("failed"):
        issues.append(f"manifest failed={manifest['failed']}")
    if len(paths) != args.expected_files:
        issues.append(f"ROOT files={len(paths)}")
    missing_outputs = sorted(expected_names - actual_names)
    extra_outputs = sorted(actual_names - expected_names)
    if missing_outputs:
        issues.append(f"missing outputs={missing_outputs[:3]}")
    if extra_outputs:
        issues.append(f"extra outputs={extra_outputs[:3]}")

    seeds = list(manifest_seeds.values())
    if len(seeds) != args.expected_files:
        issues.append(f"manifest seeds={len(seeds)}")
    if len(set(seeds)) != len(seeds):
        issues.append("digitization seeds are not unique")
    if any(not isinstance(seed, int) or seed <= 0 for seed in seeds):
        issues.append("digitization seeds must be positive integers")

    sidecar_failures = []
    for input_name, seed in manifest_seeds.items():
        stem = Path(input_name).stem
        status_path = args.log_dir / f"{stem}.status"
        seed_path = args.log_dir / f"{stem}.seed"
        if not status_path.is_file() or status_path.read_text().strip() != "0":
            sidecar_failures.append(f"{stem}: status")
        if not seed_path.is_file() or seed_path.read_text().strip() != str(seed):
            sidecar_failures.append(f"{stem}: seed")
    if sidecar_failures:
        issues.append(f"sidecar failures={sidecar_failures[:3]}")

    split_events = Counter()
    label_events = Counter()
    source_events = Counter()
    n_zero = Counter()
    n_one = Counter()
    nonfinite_energy_events = Counter()
    total_mc = 0
    total_rec = 0

    for index, path in enumerate(paths, 1):
        match = NAME_PATTERN.match(path.name)
        if match is None:
            issues.append(f"unexpected file name: {path.name}")
            continue
        split, particle, raw_source, _, _ = match.groups()
        source_id = int(raw_source)
        label = 0 if particle == "antip" else 1

        try:
            with uproot.open(path) as root_file:
                for tree_name in ("TreeMc", "TreeRec"):
                    if tree_name not in root_file:
                        raise RuntimeError(f"missing {tree_name}")
                mc_entries = int(root_file["TreeMc"].num_entries)
                rec = root_file["TreeRec"]
                rec_entries = int(rec.num_entries)
                if mc_entries != rec_entries:
                    raise RuntimeError(
                        f"TreeMc={mc_entries}, TreeRec={rec_entries}"
                    )
                for branch in REQUIRED_HITSERIES:
                    if branch not in rec:
                        raise RuntimeError(f"missing {branch}")

                file_zero = 0
                file_one = 0
                file_bad_energy = 0
                for start in range(0, rec_entries, args.chunk_size):
                    stop = min(start + args.chunk_size, rec_entries)
                    volume = rec[VOLUME_BRANCH].array(
                        entry_start=start, entry_stop=stop, library="ak"
                    )
                    energy = rec[ENERGY_BRANCH].array(
                        entry_start=start, entry_stop=stop, library="ak"
                    )
                    volume_count = ak.to_numpy(ak.num(volume, axis=1))
                    energy_count = ak.to_numpy(ak.num(energy, axis=1))
                    if not np.array_equal(volume_count, energy_count):
                        raise RuntimeError("volume/energy hit counts differ")
                    file_zero += int(np.count_nonzero(volume_count == 0))
                    file_one += int(np.count_nonzero(volume_count == 1))
                    bad_per_event = ak.to_numpy(
                        ak.sum(~np.isfinite(energy), axis=1)
                    )
                    file_bad_energy += int(np.count_nonzero(bad_per_event))

                total_mc += mc_entries
                total_rec += rec_entries
                split_events[split] += rec_entries
                label_events[label] += rec_entries
                source_events[source_id] += rec_entries
                n_zero[split] += file_zero
                n_one[split] += file_one
                nonfinite_energy_events[split] += file_bad_energy
        except Exception as error:
            issues.append(f"{path.name}: {error}")
        print(f"[{index:02d}/{len(paths):02d}] {path.name}", flush=True)

    if total_mc != args.expected_events:
        issues.append(f"TreeMc total={total_mc}")
    if total_rec != args.expected_events:
        issues.append(f"TreeRec total={total_rec}")
    if dict(split_events) != expected_splits:
        issues.append(f"split events={dict(split_events)}")
    expected_per_label = args.expected_events // 2
    if dict(label_events) != {0: expected_per_label, 1: expected_per_label}:
        issues.append(f"label events={dict(label_events)}")
    if sum(nonfinite_energy_events.values()) != 0:
        issues.append(
            f"non-finite energy events={dict(nonfinite_energy_events)}"
        )

    fallback_counts, fatal_counts = count_log_patterns(args.log_dir)
    if sum(fallback_counts.values()) != 0:
        issues.append(f"tracker fallback warnings={dict(fallback_counts)}")
    if sum(fatal_counts.values()) != 0:
        issues.append(f"fatal log patterns={dict(fatal_counts)}")

    summary = {
        "digitized_dir": str(args.digitized_dir.resolve()),
        "files": len(paths),
        "TreeMc": total_mc,
        "TreeRec": total_rec,
        "split_events": dict(sorted(split_events.items())),
        "label_events": dict(sorted(label_events.items())),
        "source_events": dict(sorted(source_events.items())),
        "n_zero_hits": dict(sorted(n_zero.items())),
        "n_one_hit": dict(sorted(n_one.items())),
        "n_le_1_hits": {
            split: n_zero[split] + n_one[split]
            for split in ("train", "val", "test")
        },
        "nonfinite_energy_events": dict(
            sorted(nonfinite_energy_events.items())
        ),
        "digitization_seed_base": manifest.get("digitization_seed_base"),
        "digitization_seeds": len(seeds),
        "unique_digitization_seeds": len(set(seeds)),
        "tracker_fallback_warnings": dict(fallback_counts),
        "fatal_log_patterns": dict(fatal_counts),
        "issues": issues,
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved: {args.output}")
    if issues:
        raise SystemExit(1)
    print("OLD TREEMC 4M DIGITIZATION AUDIT: VALID")


if __name__ == "__main__":
    main()

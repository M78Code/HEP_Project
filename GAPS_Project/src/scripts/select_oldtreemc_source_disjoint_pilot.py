#!/usr/bin/env python3
"""Select source-disjoint old TreeMc events for a small TreeRec pilot.

The 2021 Nakagami CSV stores ``source_id`` and the global TreeMc chain entry
in its first two columns.  This script selects exact entries for independent
training and validation simulation sources, discovers their ROOT files, and
writes entry lists accepted by ``tools/export/skim_treemc_entries.cc``.

The existing validation-source paired sample is intentionally not selected
here; it remains an untouched test set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Selection:
    split: str
    particle: str
    source_id: int
    label: int
    count: int

    @property
    def name(self) -> str:
        return f"{self.split}_{self.particle}_{self.source_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-events-per-class", type=int, default=1000)
    parser.add_argument("--val-events-per-class", type=int, default=500)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "repeatable prior selection manifest; entries from matching "
            "sources are excluded from the new selection"
        ),
    )
    parser.add_argument("--train-antip-source", type=int, default=1627528606)
    parser.add_argument("--train-antid-source", type=int, default=1627550259)
    parser.add_argument("--val-antip-source", type=int, default=1627528610)
    parser.add_argument("--val-antid-source", type=int, default=1627550262)
    parser.add_argument(
        "--selection",
        action="append",
        metavar="SPLIT:PARTICLE:SOURCE_ID:LABEL:COUNT",
        help=(
            "repeatable explicit selection; when supplied, replaces the "
            "four default pilot selections"
        ),
    )
    return parser.parse_args()


def selections_from_args(args: argparse.Namespace) -> list[Selection]:
    if args.selection:
        selections = []
        for value in args.selection:
            fields = value.split(":")
            if len(fields) != 5:
                raise ValueError(
                    "--selection must be "
                    "SPLIT:PARTICLE:SOURCE_ID:LABEL:COUNT"
                )
            split, particle, source_id, label, count = fields
            if split not in {"train", "val"}:
                raise ValueError(f"invalid selection split: {split}")
            if particle not in {"antip", "antid"}:
                raise ValueError(f"invalid selection particle: {particle}")
            expected_label = 0 if particle == "antip" else 1
            if int(label) != expected_label:
                raise ValueError(
                    f"{particle} selection must use label {expected_label}"
                )
            if int(count) < 1:
                raise ValueError("selection count must be positive")
            selections.append(
                Selection(
                    split,
                    particle,
                    int(source_id),
                    int(label),
                    int(count),
                )
            )
        return selections

    return [
        Selection(
            "train", "antip", args.train_antip_source, 0,
            args.train_events_per_class,
        ),
        Selection(
            "train", "antid", args.train_antid_source, 1,
            args.train_events_per_class,
        ),
        Selection(
            "val", "antip", args.val_antip_source, 0,
            args.val_events_per_class,
        ),
        Selection(
            "val", "antid", args.val_antid_source, 1,
            args.val_events_per_class,
        ),
    ]


def discover_root_files(root_dir: Path, source_id: int) -> list[Path]:
    paths = list(root_dir.glob(f"*_{source_id}.root"))
    paths.extend(root_dir.glob(f"*_{source_id}_*.root"))
    paths = sorted(
        set(paths),
        key=lambda path: (
            not path.stem.endswith(f"_{source_id}"),
            path.name,
        ),
    )
    if not paths:
        raise FileNotFoundError(
            f"no ROOT file for source {source_id} under {root_dir}"
        )
    return paths


def load_excluded_entries(
    manifest_paths: list[Path],
) -> dict[int, set[int]]:
    excluded: dict[int, set[int]] = {}
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in payload["selections"]:
            source_id = int(row["source_id"])
            entry_path = Path(row["entry_list"])
            if not entry_path.is_file():
                recovered_path = manifest_path.parent / entry_path.name
                if recovered_path.is_file():
                    entry_path = recovered_path
                else:
                    raise FileNotFoundError(
                        f"excluded entry list is missing: {entry_path}"
                    )
            entries = {
                int(line)
                for line in entry_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            expected = int(row["events"])
            if len(entries) != expected:
                raise RuntimeError(
                    f"{entry_path}: expected {expected} unique entries, "
                    f"found {len(entries)}"
                )
            excluded.setdefault(source_id, set()).update(entries)

    if manifest_paths:
        print(
            "excluded existing entries:",
            {key: len(value) for key, value in sorted(excluded.items())},
            flush=True,
        )
    return excluded


def scan_csv(
    csv_dir: Path,
    selections: list[Selection],
    excluded: dict[int, set[int]],
) -> tuple[dict[str, list[int]], list[Path]]:
    selected = {item.name: [] for item in selections}
    by_source = {item.source_id: item for item in selections}
    scanned_paths: list[Path] = []

    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no CSV files under {csv_dir}")

    for csv_path in csv_paths:
        scanned_paths.append(csv_path)
        with csv_path.open() as stream:
            for row_number, line in enumerate(stream, 1):
                prefix = line.split(",", 3)
                if len(prefix) < 3:
                    continue
                try:
                    source_id = int(prefix[0])
                except ValueError:
                    continue
                item = by_source.get(source_id)
                if item is None or len(selected[item.name]) >= item.count:
                    continue
                try:
                    source_entry = int(prefix[1])
                    label = int(prefix[2])
                except ValueError as error:
                    raise RuntimeError(
                        f"invalid target row {csv_path}:{row_number}"
                    ) from error
                if label != item.label:
                    raise RuntimeError(
                        f"label mismatch for source {source_id}: "
                        f"expected {item.label}, found {label} at "
                        f"{csv_path}:{row_number}"
                    )
                if source_entry in excluded.get(source_id, set()):
                    continue
                selected[item.name].append(source_entry)

        progress = ", ".join(
            f"{item.name}={len(selected[item.name])}/{item.count}"
            for item in selections
        )
        print(f"scanned {csv_path.name}: {progress}", flush=True)
        if all(len(selected[item.name]) == item.count for item in selections):
            break

    return selected, scanned_paths


def validate_entries(
    selections: list[Selection],
    selected: dict[str, list[int]],
    excluded: dict[int, set[int]],
) -> None:
    source_ids = [item.source_id for item in selections]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("train/val source IDs must all be distinct")
    test_source_ids = {1627528714, 1627550286}
    overlap_with_test = sorted(set(source_ids) & test_source_ids)
    if overlap_with_test:
        raise ValueError(
            f"train/val source IDs overlap untouched test: {overlap_with_test}"
        )

    split_entries: dict[str, set[tuple[int, int]]] = {"train": set(), "val": set()}
    for item in selections:
        entries = selected[item.name]
        if len(entries) != item.count:
            raise RuntimeError(
                f"{item.name}: expected {item.count} events, found {len(entries)}"
            )
        if len(set(entries)) != len(entries):
            raise RuntimeError(f"{item.name}: duplicate source entries")
        overlap_with_existing = set(entries) & excluded.get(item.source_id, set())
        if overlap_with_existing:
            raise RuntimeError(
                f"{item.name}: overlaps excluded selection: "
                f"{sorted(overlap_with_existing)[:5]}"
            )
        split_entries[item.split].update(
            (item.source_id, entry) for entry in entries
        )

    overlap = split_entries["train"] & split_entries["val"]
    if overlap:
        raise RuntimeError(f"train/val event overlap: {sorted(overlap)[:5]}")


def write_outputs(
    args: argparse.Namespace,
    selections: list[Selection],
    selected: dict[str, list[int]],
    scanned_paths: list[Path],
) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in selections:
        entries = sorted(selected[item.name])
        entry_path = args.output_dir / f"{item.name}.entries"
        entry_path.write_text(
            "".join(f"{entry}\n" for entry in entries),
            encoding="utf-8",
        )
        root_files = discover_root_files(args.root_dir, item.source_id)
        rows.append(
            {
                "split": item.split,
                "particle": item.particle,
                "label": item.label,
                "source_id": item.source_id,
                "events": len(entries),
                "entry_min": min(entries),
                "entry_max": max(entries),
                "entry_list": str(entry_path.resolve()),
                "root_files": [str(path.resolve()) for path in root_files],
                "metadata_file": str(root_files[0].resolve()),
            }
        )
        print(
            f"{item.name}: events={len(entries):,} "
            f"range=[{min(entries):,}, {max(entries):,}] "
            f"ROOT files={len(root_files)}",
            flush=True,
        )

    manifest = {
        "purpose": "old-domain source-disjoint TreeRec pilot",
        "csv_dir": str(args.csv_dir.resolve()),
        "root_dir": str(args.root_dir.resolve()),
        "excluded_selection_manifests": [
            str(path.resolve()) for path in args.exclude_manifest
        ],
        "scanned_csv_files": [str(path.resolve()) for path in scanned_paths],
        "untouched_test_sources": {
            "antip": 1627528714,
            "antid": 1627550286,
        },
        "selections": rows,
    }
    manifest_path = args.output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved: {manifest_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.train_events_per_class < 1 or args.val_events_per_class < 1:
        raise ValueError("event counts must be positive")
    selections = selections_from_args(args)
    excluded = load_excluded_entries(args.exclude_manifest)
    selected, scanned_paths = scan_csv(args.csv_dir, selections, excluded)
    validate_entries(selections, selected, excluded)
    write_outputs(args, selections, selected, scanned_paths)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select an arbitrary source-disjoint old-TreeMc train/val/test dataset."""

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
    parser.add_argument(
        "--csv-dir",
        action="append",
        required=True,
        metavar="SPLIT=PATH",
        help="repeatable CSV directory mapping for train, val, or test",
    )
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        action="append",
        required=True,
        metavar="SPLIT:PARTICLE:SOURCE_ID:LABEL:COUNT",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="repeatable prior manifest whose selected entries are excluded",
    )
    parser.add_argument(
        "--purpose",
        default="old-domain source-disjoint TreeRec dataset",
    )
    return parser.parse_args()


def parse_csv_dirs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        split, separator, raw_path = value.partition("=")
        if not separator or split not in {"train", "val", "test"}:
            raise ValueError("--csv-dir must be SPLIT=PATH")
        if split in result:
            raise ValueError(f"duplicate CSV directory for {split}")
        path = Path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(f"missing {split} CSV directory: {path}")
        result[split] = path
    return result


def parse_selections(values: list[str]) -> list[Selection]:
    result = []
    names = set()
    for value in values:
        fields = value.split(":")
        if len(fields) != 5:
            raise ValueError(
                "--selection must be SPLIT:PARTICLE:SOURCE_ID:LABEL:COUNT"
            )
        split, particle, source_id, label, count = fields
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split: {split}")
        if particle not in {"antip", "antid"}:
            raise ValueError(f"invalid particle: {particle}")
        expected_label = 0 if particle == "antip" else 1
        if int(label) != expected_label:
            raise ValueError(f"{particle} must use label {expected_label}")
        selection = Selection(
            split=split,
            particle=particle,
            source_id=int(source_id),
            label=int(label),
            count=int(count),
        )
        if selection.count < 1:
            raise ValueError("selection count must be positive")
        if selection.name in names:
            raise ValueError(f"duplicate selection: {selection.name}")
        names.add(selection.name)
        result.append(selection)

    source_splits: dict[int, str] = {}
    for selection in result:
        prior = source_splits.setdefault(selection.source_id, selection.split)
        if prior != selection.split:
            raise ValueError(
                f"source {selection.source_id} occurs in {prior} and "
                f"{selection.split}"
            )
    return result


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


def load_excluded_entries(manifests: list[Path]) -> dict[int, set[int]]:
    excluded: dict[int, set[int]] = {}
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for row in payload["selections"]:
            entry_path = Path(row["entry_list"])
            if not entry_path.is_file():
                candidate = manifest.parent / entry_path.name
                if not candidate.is_file():
                    raise FileNotFoundError(f"missing entry list: {entry_path}")
                entry_path = candidate
            entries = {
                int(line)
                for line in entry_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            expected = int(row["events"])
            if len(entries) != expected:
                raise RuntimeError(
                    f"{entry_path}: expected {expected}, found {len(entries)}"
                )
            excluded.setdefault(int(row["source_id"]), set()).update(entries)
    return excluded


def scan_split(
    csv_dir: Path,
    selections: list[Selection],
    excluded: dict[int, set[int]],
) -> tuple[dict[str, list[int]], list[Path]]:
    selected = {item.name: [] for item in selections}
    by_source = {item.source_id: item for item in selections}
    scanned = []
    paths = sorted(csv_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no CSV files under {csv_dir}")

    for path in paths:
        scanned.append(path)
        with path.open() as stream:
            for row_number, line in enumerate(stream, 1):
                fields = line.split(",", 3)
                if len(fields) < 3:
                    continue
                try:
                    source_id = int(fields[0])
                except ValueError:
                    continue
                item = by_source.get(source_id)
                if item is None or len(selected[item.name]) >= item.count:
                    continue
                try:
                    source_entry = int(fields[1])
                    label = int(fields[2])
                except ValueError as error:
                    raise RuntimeError(f"invalid row {path}:{row_number}") from error
                if label != item.label:
                    raise RuntimeError(
                        f"label mismatch for source {source_id} at "
                        f"{path}:{row_number}"
                    )
                if source_entry in excluded.get(source_id, set()):
                    continue
                selected[item.name].append(source_entry)

        progress = ", ".join(
            f"{item.name}={len(selected[item.name]):,}/{item.count:,}"
            for item in selections
        )
        print(f"scanned {path.name}: {progress}", flush=True)
        if all(len(selected[item.name]) == item.count for item in selections):
            break
    return selected, scanned


def main() -> None:
    args = parse_args()
    csv_dirs = parse_csv_dirs(args.csv_dir)
    selections = parse_selections(args.selection)
    missing_csv_splits = sorted({item.split for item in selections} - csv_dirs.keys())
    if missing_csv_splits:
        raise ValueError(f"missing --csv-dir for: {missing_csv_splits}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    excluded = load_excluded_entries(args.exclude_manifest)
    selected: dict[str, list[int]] = {}
    scanned_by_split = {}
    for split in ("train", "val", "test"):
        split_selections = [item for item in selections if item.split == split]
        if not split_selections:
            continue
        split_selected, scanned = scan_split(
            csv_dirs[split], split_selections, excluded
        )
        selected.update(split_selected)
        scanned_by_split[split] = [str(path.resolve()) for path in scanned]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in selections:
        entries = selected[item.name]
        if len(entries) != item.count:
            raise RuntimeError(
                f"{item.name}: expected {item.count:,}, found {len(entries):,}"
            )
        if len(set(entries)) != len(entries):
            raise RuntimeError(f"{item.name}: duplicate source entries")
        if set(entries) & excluded.get(item.source_id, set()):
            raise RuntimeError(f"{item.name}: overlaps excluded entries")
        entries = sorted(entries)
        entry_path = args.output_dir / f"{item.name}.entries"
        entry_path.write_text(
            "".join(f"{entry}\n" for entry in entries), encoding="utf-8"
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
            f"range=[{min(entries):,}, {max(entries):,}]",
            flush=True,
        )

    summary = {
        split: {
            "events": sum(item.count for item in selections if item.split == split),
            "sources": sorted(
                item.source_id for item in selections if item.split == split
            ),
        }
        for split in ("train", "val", "test")
    }
    manifest = {
        "purpose": args.purpose,
        "csv_dirs": {key: str(value.resolve()) for key, value in csv_dirs.items()},
        "root_dir": str(args.root_dir.resolve()),
        "excluded_selection_manifests": [
            str(path.resolve()) for path in args.exclude_manifest
        ],
        "scanned_csv_files": scanned_by_split,
        "summary": summary,
        "selections": rows,
    }
    destination = args.output_dir / "selection_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved: {destination}", flush=True)


if __name__ == "__main__":
    main()

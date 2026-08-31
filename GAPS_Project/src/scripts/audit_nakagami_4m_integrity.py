#!/usr/bin/env python3
"""Audit provenance and split integrity of the old Nakagami fixed-grid data.

The audit is deliberately training-free.  It checks the exported arrays and,
when the source CSV directory is available, streams only the first six columns
needed for event identity, label, beta, category, and stopping layer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
PARTICLE_RE = re.compile(r"_(Dbar|Pbar)_", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--csv-dir", type=Path)
    parser.add_argument("--glob", default="CNN*Atrest*.csv")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--reservoir-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.dataset_dir is None and args.csv_dir is None:
        parser.error("at least one of --dataset-dir or --csv-dir is required")
    return args


def scalar_summary(values: np.ndarray) -> dict[str, object]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": int(values.size), "finite": 0}
    quantiles = np.quantile(finite, [0.01, 0.1, 0.5, 0.9, 0.99])
    return {
        "n": int(values.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "q01": float(quantiles[0]),
        "q10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q90": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def audit_export(dataset_dir: Path) -> dict[str, object]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
    split_dirs = [
        dataset_dir / f"{split}_nakagami_style_4M" for split in SPLITS
    ]
    if not any(path.is_dir() for path in split_dirs):
        raise FileNotFoundError(
            f"no *_nakagami_style_4M split directories found under {dataset_dir}"
        )
    report: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "splits": {},
    }
    manifest_path = dataset_dir / "export_manifest.json"
    summary_path = dataset_dir / "export_summary.json"
    metadata = None
    for candidate in (manifest_path, summary_path):
        if candidate.exists():
            metadata = json.loads(candidate.read_text())
            report["export_metadata_path"] = str(candidate)
            report["export_metadata"] = metadata
            break
    if metadata is None:
        report["export_metadata_path"] = None

    source_files_by_split: dict[str, set[str]] = {}
    if isinstance(metadata, dict) and metadata.get("split_meta"):
        for split, item in metadata["split_meta"].items():
            source_files_by_split[split] = set(item.get("files", []))
    elif isinstance(metadata, dict) and metadata.get("explicit_split") is False:
        inputs = set(metadata.get("inputs", []))
        source_files_by_split = {split: inputs for split in SPLITS}

    if source_files_by_split:
        report["source_file_counts_by_split"] = {
            split: len(files) for split, files in source_files_by_split.items()
        }
        report["source_file_overlap"] = {
            f"{left}_vs_{right}": len(
                source_files_by_split.get(left, set())
                & source_files_by_split.get(right, set())
            )
            for i, left in enumerate(SPLITS)
            for right in SPLITS[i + 1 :]
        }

    for split in SPLITS:
        split_dir = dataset_dir / f"{split}_nakagami_style_4M"
        if not split_dir.exists():
            report["splits"][split] = {"missing": True}
            continue
        item: dict[str, object] = {"directory": str(split_dir)}
        arrays = {}
        for name in ("voxels", "tof_paddles", "tof_primary", "labels", "betas"):
            path = split_dir / f"{name}.npy"
            if not path.exists():
                arrays[name] = {"missing": True}
                continue
            array = np.load(path, mmap_mode="r")
            arrays[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
            if name == "labels":
                labels, counts = np.unique(array, return_counts=True)
                arrays[name]["counts"] = {
                    str(int(label)): int(count)
                    for label, count in zip(labels, counts)
                }
            elif name == "betas":
                arrays[name]["summary"] = scalar_summary(np.asarray(array))
            elif name == "tof_paddles":
                arrays[name]["nonzero_fraction"] = float(np.count_nonzero(array) / array.size)
        item["arrays"] = arrays
        report["splits"][split] = item
    return report


def particle_from_name(path: Path) -> int | None:
    match = PARTICLE_RE.search(path.name)
    if match is None:
        return None
    return 1 if match.group(1).lower() == "dbar" else 0


def reservoir_add(
    values: list[float], value: float, seen: int, capacity: int, rng: random.Random
) -> None:
    if capacity <= 0:
        return
    if len(values) < capacity:
        values.append(value)
        return
    index = rng.randrange(seen)
    if index < capacity:
        values[index] = value


def audit_csv(args: argparse.Namespace) -> dict[str, object]:
    if args.csv_dir is None:
        return {"skipped": True, "reason": "--csv-dir was not supplied"}
    files = sorted(args.csv_dir.glob(args.glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(
            f"no source CSV matched {args.glob!r} under {args.csv_dir}"
        )

    rng = random.Random(args.seed)
    seen_keys: dict[int, int] = {}
    duplicate_within_file = 0
    duplicate_across_files = 0
    duplicate_cross_label = 0
    rows = 0
    malformed = 0
    filename_label_mismatch = 0
    source_id_filename_mismatch = 0
    category_counts: Counter[int] = Counter()
    stopping_layer_counts: Counter[int] = Counter()
    rows_by_label: Counter[int] = Counter()
    files_by_label: Counter[int] = Counter()
    beta_reservoir = {0: [], 1: []}
    beta_seen = Counter()
    file_summaries = []

    for file_index, path in enumerate(files):
        expected_label = particle_from_name(path)
        if expected_label is not None:
            files_by_label[expected_label] += 1
        rows_this_file = 0
        duplicate_this_file = 0
        with path.open(errors="replace") as handle:
            for line_index, line in enumerate(handle):
                if args.max_rows_per_file > 0 and line_index >= args.max_rows_per_file:
                    break
                prefix = line.rstrip("\n").split(",", 6)
                if len(prefix) < 6:
                    malformed += 1
                    continue
                try:
                    source_id = int(float(prefix[0]))
                    event_id = int(float(prefix[1]))
                    label = int(float(prefix[2]))
                    category = int(float(prefix[3]))
                    beta = float(prefix[4])
                    stopping_layer = int(float(prefix[5]))
                except (TypeError, ValueError, OverflowError):
                    malformed += 1
                    continue
                if label not in (0, 1) or not math.isfinite(beta):
                    malformed += 1
                    continue

                rows += 1
                rows_this_file += 1
                rows_by_label[label] += 1
                category_counts[category] += 1
                stopping_layer_counts[stopping_layer] += 1
                beta_seen[label] += 1
                reservoir_add(
                    beta_reservoir[label],
                    beta,
                    beta_seen[label],
                    args.reservoir_size,
                    rng,
                )
                if expected_label is not None and label != expected_label:
                    filename_label_mismatch += 1

                match = re.search(r"(\d{10})(?:_\d+)?\.csv$", path.name)
                if match and source_id != int(match.group(1)):
                    source_id_filename_mismatch += 1

                key = (source_id << 32) ^ (event_id & 0xFFFFFFFF)
                previous = seen_keys.get(key)
                encoded = (file_index << 1) | label
                if previous is not None:
                    previous_file = previous >> 1
                    previous_label = previous & 1
                    if previous_file == file_index:
                        duplicate_within_file += 1
                        duplicate_this_file += 1
                    else:
                        duplicate_across_files += 1
                    if previous_label != label:
                        duplicate_cross_label += 1
                else:
                    seen_keys[key] = encoded

        file_summaries.append(
            {
                "file": str(path),
                "rows": rows_this_file,
                "duplicates_within_file": duplicate_this_file,
                "expected_label": expected_label,
            }
        )

    return {
        "csv_dir": str(args.csv_dir),
        "glob": args.glob,
        "files": len(files),
        "files_by_label": {str(k): int(v) for k, v in files_by_label.items()},
        "rows": rows,
        "rows_by_label": {str(k): int(v) for k, v in rows_by_label.items()},
        "malformed_rows": malformed,
        "filename_label_mismatch": filename_label_mismatch,
        "source_id_filename_mismatch": source_id_filename_mismatch,
        "unique_event_keys": len(seen_keys),
        "duplicate_within_file": duplicate_within_file,
        "duplicate_across_files": duplicate_across_files,
        "duplicate_cross_label": duplicate_cross_label,
        "category_counts": {str(k): int(v) for k, v in category_counts.items()},
        "stopping_layer_counts": {
            str(k): int(v) for k, v in stopping_layer_counts.items()
        },
        "beta_by_label": {
            str(label): scalar_summary(np.asarray(values, dtype=np.float64))
            for label, values in beta_reservoir.items()
        },
        "file_summaries": file_summaries,
        "note": (
            "The no-beta fixed-grid model excludes columns 0:6, but the grid and "
            "TOF/global columns were produced from TreeMc tracks after stopped/top-trigger "
            "selection; they are not TreeRec observables."
        ),
    }


def print_findings(report: dict[str, object]) -> None:
    export = report.get("export")
    csv_report = report["source_csv"]
    print("Nakagami 4M integrity audit")
    metadata = None
    if isinstance(export, dict):
        print("dataset:", export["dataset_dir"])
        metadata = export.get("export_metadata")
        if isinstance(metadata, dict):
            print("layout:", metadata.get("layout"))
            print("explicit split:", metadata.get("explicit_split"))
        if export.get("source_file_overlap"):
            print("source file overlap:", export["source_file_overlap"])
        for split, item in export["splits"].items():
            if item.get("missing"):
                print(split, "MISSING")
                continue
            labels = item["arrays"].get("labels", {}).get("counts")
            betas = item["arrays"].get("betas", {}).get("summary")
            print(split, "labels=", labels, "beta=", betas)
    if not csv_report.get("skipped"):
        print("source files:", csv_report["files"], "rows:", csv_report["rows"])
        print("rows by label:", csv_report["rows_by_label"])
        print(
            "duplicates within/across/cross-label:",
            csv_report["duplicate_within_file"],
            csv_report["duplicate_across_files"],
            csv_report["duplicate_cross_label"],
        )
        print("beta by label:", csv_report["beta_by_label"])
    print()
    print("Interpretation:")
    if isinstance(metadata, dict) and metadata.get("explicit_split") is False:
        print("- The exported train/val/test split is event-random, not source-file grouped.")
        print("- A file-grouped re-export is required to test production-file leakage.")
    print("- The fixed grid and TOF summaries are derived from TreeMc track truth.")
    print("- Their performance is not an information ceiling for TreeRec-only inference.")


def main() -> None:
    args = parse_args()
    report = {"source_csv": audit_csv(args)}
    if args.dataset_dir is not None:
        report["export"] = audit_export(args.dataset_dir)
    print_findings(report)
    output = args.output
    if output is None:
        if args.dataset_dir is not None:
            output = args.dataset_dir / "priority1_integrity_audit.json"
        else:
            output = Path("priority1_integrity_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved:", output)


if __name__ == "__main__":
    main()

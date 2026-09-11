#!/usr/bin/env python3
"""Build balanced fixed-grid train/val/test arrays from two direct exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARRAY_SPECS = {
    "voxels": (np.float32, (10, 12, 12)),
    "tof_primary": (np.float32, (11,)),
    "labels": (np.int64, ()),
    "betas": (np.float32, ()),
    "random_seeds": (np.int64, ()),
    "chain_entries": (np.int64, ()),
    "source_file_indices": (np.int32, ()),
    "source_entries": (np.int64, ()),
}


def load_export(directory: Path, expected_label: int) -> dict[str, np.ndarray]:
    if not (directory / "_SUCCESS").is_file():
        raise RuntimeError(f"incomplete direct export: {directory}")

    arrays: dict[str, np.ndarray] = {}
    event_count = None
    for name, (dtype, trailing_shape) in ARRAY_SPECS.items():
        path = directory / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        if array.dtype != np.dtype(dtype):
            raise TypeError(f"{path}: dtype {array.dtype} != {np.dtype(dtype)}")
        if array.shape[1:] != trailing_shape:
            raise ValueError(f"{path}: trailing shape {array.shape[1:]} != {trailing_shape}")
        if event_count is None:
            event_count = len(array)
        elif len(array) != event_count:
            raise ValueError(f"{path}: event count {len(array)} != {event_count}")
        arrays[name] = array

    labels = np.asarray(arrays["labels"])
    if not np.all(labels == expected_label):
        values, counts = np.unique(labels, return_counts=True)
        raise ValueError(
            f"{directory}: expected label {expected_label}, got "
            f"{dict(zip(values.tolist(), counts.tolist()))}"
        )
    return arrays


def allocate_split(directory: Path, event_count: int) -> dict[str, np.memmap]:
    directory.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.memmap] = {}
    for name, (dtype, trailing_shape) in ARRAY_SPECS.items():
        arrays[name] = np.lib.format.open_memmap(
            directory / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(event_count, *trailing_shape),
        )
    arrays["tof_paddles"] = np.lib.format.open_memmap(
        directory / "tof_paddles.npy",
        mode="w+",
        dtype=np.float32,
        shape=(event_count, 172),
    )
    arrays["source_particle"] = np.lib.format.open_memmap(
        directory / "source_particle.npy",
        mode="w+",
        dtype=np.int8,
        shape=(event_count,),
    )
    return arrays


def copy_interleaved(
    destination: dict[str, np.memmap],
    antip: dict[str, np.ndarray],
    antid: dict[str, np.ndarray],
    start: int,
    stop: int,
    chunk_size: int,
) -> None:
    output_cursor = 0
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        count = chunk_stop - chunk_start
        even = slice(output_cursor, output_cursor + 2 * count, 2)
        odd = slice(output_cursor + 1, output_cursor + 2 * count, 2)
        source_slice = slice(chunk_start, chunk_stop)

        for name in ARRAY_SPECS:
            destination[name][even] = antip[name][source_slice]
            destination[name][odd] = antid[name][source_slice]
        destination["tof_paddles"][output_cursor : output_cursor + 2 * count] = 0.0
        destination["source_particle"][even] = 0
        destination["source_particle"][odd] = 1
        output_cursor += 2 * count

    for array in destination.values():
        array.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--antip-dir", type=Path, required=True)
    parser.add_argument("--antid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, required=True)
    parser.add_argument("--val-per-class", type=int, required=True)
    parser.add_argument("--test-per-class", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(
        args.train_per_class,
        args.val_per_class,
        args.test_per_class,
        args.chunk_size,
    ) <= 0:
        raise ValueError("all event counts and chunk size must be positive")

    antip = load_export(args.antip_dir, 0)
    antid = load_export(args.antid_dir, 1)
    required_per_class = (
        args.train_per_class + args.val_per_class + args.test_per_class
    )
    if len(antip["labels"]) < required_per_class:
        raise ValueError(
            f"antiP export has {len(antip['labels']):,} events, "
            f"need {required_per_class:,}"
        )
    if len(antid["labels"]) < required_per_class:
        raise ValueError(
            f"antiD export has {len(antid['labels']):,} events, "
            f"need {required_per_class:,}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_counts = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }
    split_summary = {}
    cursor = 0
    for split, per_class in split_counts.items():
        split_dir = args.output_dir / f"{split}_nakagami_style_4M"
        output = allocate_split(split_dir, 2 * per_class)
        copy_interleaved(
            output,
            antip,
            antid,
            cursor,
            cursor + per_class,
            args.chunk_size,
        )
        split_summary[split] = {
            "events": 2 * per_class,
            "events_per_class": per_class,
            "source_range": [cursor, cursor + per_class],
            "label_counts": {"0": per_class, "1": per_class},
        }
        print(
            f"[{split}] saved {2 * per_class:,} events "
            f"({per_class:,} per class)",
            flush=True,
        )
        cursor += per_class

    for particle, source in (("antiP", args.antip_dir), ("antiD", args.antid_dir)):
        source_list = source / "source_files.txt"
        if source_list.is_file():
            (args.output_dir / f"{particle}_source_files.txt").write_text(
                source_list.read_text(encoding="utf-8"), encoding="utf-8"
            )

    manifest = {
        "source": "direct TreeMc fixed-grid binary dataset",
        "antip_dir": str(args.antip_dir),
        "antid_dir": str(args.antid_dir),
        "input_features": {
            "voxel_shape": [10, 12, 12],
            "tof_primary": 11,
            "tof_paddles": 172,
            "tof_paddles_value": 0.0,
            "beta_is_model_input": False,
        },
        "splits": split_summary,
        "events": 2 * required_per_class,
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").write_text("ok\n", encoding="ascii")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

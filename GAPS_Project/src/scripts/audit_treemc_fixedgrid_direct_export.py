#!/usr/bin/env python3
"""Compare direct TreeMc NPY output with the legacy CSV conversion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


N_VOXELS = 10 * 12 * 12
N_TOF_PRIMARY = 11


def load_legacy_csv(path: Path) -> dict[str, np.ndarray]:
    voxels = []
    tof_primary = []
    labels = []
    betas = []
    random_seeds = []
    chain_entries = []

    with path.open(newline="") as handle:
        for row_index, row in enumerate(csv.reader(handle)):
            if len(row) != 1457:
                raise ValueError(
                    f"{path}: row {row_index} has {len(row)} columns, expected 1457"
                )
            voxel = np.asarray(row[6:1446], dtype=np.float32)
            tof = np.asarray(row[1446:1457], dtype=np.float32)
            if voxel.size != N_VOXELS or tof.size != N_TOF_PRIMARY:
                raise ValueError(f"{path}: malformed row {row_index}")

            voxels.append(voxel.reshape(10, 12, 12))
            tof_primary.append(tof)
            labels.append(int(row[2]))
            betas.append(np.float32(row[4]))
            random_seeds.append(int(row[0]))
            chain_entries.append(int(row[1]))

    return {
        "voxels": np.asarray(voxels, dtype=np.float32),
        "tof_primary": np.asarray(tof_primary, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "betas": np.asarray(betas, dtype=np.float32),
        "random_seeds": np.asarray(random_seeds, dtype=np.int64),
        "chain_entries": np.asarray(chain_entries, dtype=np.int64),
    }


def load_direct_npy(path: Path) -> dict[str, np.ndarray]:
    if not (path / "_SUCCESS").is_file():
        raise FileNotFoundError(f"missing completion marker: {path / '_SUCCESS'}")
    names = (
        "voxels",
        "tof_primary",
        "labels",
        "betas",
        "random_seeds",
        "chain_entries",
    )
    return {name: np.load(path / f"{name}.npy") for name in names}


def compare(csv_values: dict[str, np.ndarray], direct: dict[str, np.ndarray]) -> None:
    if csv_values.keys() != direct.keys():
        raise AssertionError(
            f"field mismatch: CSV={sorted(csv_values)} direct={sorted(direct)}"
        )

    failures = []
    for name in csv_values:
        expected = csv_values[name]
        actual = direct[name]
        same_shape = expected.shape == actual.shape
        exact = same_shape and np.array_equal(expected, actual, equal_nan=True)

        if np.issubdtype(expected.dtype, np.floating) and same_shape:
            finite = np.isfinite(expected) & np.isfinite(actual)
            max_abs = (
                float(np.max(np.abs(expected[finite] - actual[finite])))
                if np.any(finite)
                else 0.0
            )
        else:
            max_abs = None

        print(
            f"{name:15s} shape={actual.shape!s:18s} "
            f"dtype={actual.dtype!s:7s} exact={exact} "
            f"max_abs={max_abs if max_abs is not None else '-'}"
        )
        if not exact:
            failures.append(name)

    if failures:
        raise AssertionError(f"non-equivalent fields: {failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--npy-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy = load_legacy_csv(args.csv)
    direct = load_direct_npy(args.npy_dir)
    compare(legacy, direct)
    print(f"events: {len(legacy['labels']):,}")
    print("DIRECT TREEMC FIXED-GRID EXPORT: EXACTLY EQUIVALENT")


if __name__ == "__main__":
    main()

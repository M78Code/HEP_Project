#!/usr/bin/env python3
"""Export a TreeRec graph cache to the fixed-grid CNN+DNN input format.

The exported sample contains exactly the events and split membership already
stored in the graph cache.  Si(Li) hit energies are decoded from the cache's
train-global log normalization and accumulated into a 10 x 12 x 12 grid.
The 11-D TOF vector is copied directly from ``graph.tof_feat``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from GAPS_Project.src.data_parse.voxelizer import build_sili_voxel


PARTICLES = (("antip", 0), ("antid", 1))
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-suffix", default="cnndnn_10x12x12")
    parser.add_argument("--grid-x", type=int, default=12)
    parser.add_argument("--grid-y", type=int, default=12)
    return parser.parse_args()


def count_graphs(path: Path) -> int:
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        row = json.loads(sidecar.read_text(encoding="utf-8"))
        if "n_graphs" in row:
            return int(row["n_graphs"])
    return len(torch.load(path, map_location="cpu", weights_only=False))


def load_normalizer(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    path = cache_dir / "node_feature_normalizer.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("mode") != "global_log":
        raise ValueError(f"expected global_log normalization, got {row.get('mode')!r}")
    mean = np.asarray(row["mean"], dtype=np.float64)
    std = np.asarray(row["std"], dtype=np.float64)
    if mean.shape != (6,) or std.shape != (6,) or np.any(std <= 0):
        raise ValueError(f"invalid normalizer in {path}")
    return mean, std, row


def allocate_arrays(
    directory: Path,
    event_count: int,
    grid_x: int,
    grid_y: int,
) -> dict[str, np.memmap]:
    directory.mkdir(parents=True, exist_ok=False)
    specs = {
        "voxels": (np.float32, (10, grid_x, grid_y)),
        "tof_primary": (np.float32, (11,)),
        "labels": (np.int64, ()),
        "betas": (np.float32, ()),
        "source_file_indices": (np.int64, ()),
        "source_entries": (np.int64, ()),
        "source_selected_indices": (np.int64, ()),
        "source_particle": (np.int8, ()),
        "n_hits": (np.int32, ()),
    }
    return {
        name: np.lib.format.open_memmap(
            directory / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(event_count, *shape),
        )
        for name, (dtype, shape) in specs.items()
    }


def scalar_attr(graph, name: str) -> int:
    if not hasattr(graph, name):
        raise RuntimeError(f"graph is missing provenance field {name!r}")
    return int(getattr(graph, name).view(()).item())


def decode_event(graph, mean: np.ndarray, std: np.ndarray) -> tuple[dict, float]:
    x = graph.x.detach().cpu().numpy().astype(np.float64, copy=False)
    positions = graph.pos.detach().cpu().numpy().astype(np.float32, copy=False)
    if x.ndim != 2 or x.shape[1] < 8 or positions.shape != (len(x), 3):
        raise RuntimeError(
            f"unexpected graph shapes: x={x.shape}, pos={positions.shape}"
        )

    log_energy = x[:, 3] * std[3] + mean[3]
    energies = np.expm1(log_energy)
    energies = np.clip(energies, 0.0, None).astype(np.float32)
    if not np.isfinite(energies).all() or not np.isfinite(positions).all():
        raise RuntimeError("decoded event contains non-finite values")

    detector_type = x[:, 6] > 0.5
    layer = np.rint(x[:, 7] * 16.0).astype(np.int64)
    volume_id = np.where(
        detector_type,
        (200 + layer) * 1_000_000,
        (100 + layer) * 1_000_000,
    ).astype(np.int64)

    expected_total = float(graph.total_energy.view(()).item())
    observed_total = float(energies.sum(dtype=np.float64))
    relative_error = abs(observed_total - expected_total) / max(abs(expected_total), 1.0)
    return {
        "energy": energies,
        "positions": positions,
        "volume_id": volume_id,
    }, relative_error


def export_split(
    cache_dir: Path,
    output_dir: Path,
    split: str,
    suffix: str,
    grid_x: int,
    grid_y: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    files_by_particle = {
        particle: sorted(cache_dir.glob(f"{split}_{particle}_*.pt"))
        for particle, _ in PARTICLES
    }
    for particle, files in files_by_particle.items():
        if not files:
            raise FileNotFoundError(f"no {split}_{particle}_*.pt files in {cache_dir}")

    counts = {
        particle: sum(count_graphs(path) for path in files)
        for particle, files in files_by_particle.items()
    }
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"{split}: class counts differ: {counts}")
    events_per_class = next(iter(counts.values()))
    split_dir = output_dir / f"{split}_{suffix}"
    arrays = allocate_arrays(split_dir, 2 * events_per_class, grid_x, grid_y)

    max_energy_decode_relative_error = 0.0
    try:
        for particle, expected_label in PARTICLES:
            particle_index = 0
            for path in files_by_particle[particle]:
                graphs = torch.load(path, map_location="cpu", weights_only=False)
                for graph in graphs:
                    label = int(graph.y.view(()).item())
                    if label != expected_label:
                        raise RuntimeError(
                            f"{path}: expected label {expected_label}, got {label}"
                        )
                    event, relative_error = decode_event(graph, mean, std)
                    max_energy_decode_relative_error = max(
                        max_energy_decode_relative_error, relative_error
                    )
                    output_index = 2 * particle_index + expected_label
                    arrays["voxels"][output_index] = build_sili_voxel(
                        event, grid_x=grid_x, grid_y=grid_y
                    )
                    tof = graph.tof_feat.detach().cpu().numpy().astype(
                        np.float32, copy=False
                    ).reshape(-1)
                    if tof.shape != (11,) or not np.isfinite(tof).all():
                        raise RuntimeError(f"{path}: invalid tof_feat shape or values")
                    arrays["tof_primary"][output_index] = tof
                    arrays["labels"][output_index] = label
                    arrays["betas"][output_index] = float(
                        graph.mc_beta.view(()).item()
                    )
                    arrays["source_file_indices"][output_index] = scalar_attr(
                        graph, "source_file_index"
                    )
                    arrays["source_entries"][output_index] = scalar_attr(
                        graph, "source_root_entry"
                    )
                    arrays["source_selected_indices"][output_index] = scalar_attr(
                        graph, "source_selected_index"
                    )
                    arrays["source_particle"][output_index] = expected_label
                    arrays["n_hits"][output_index] = int(graph.num_nodes)
                    particle_index += 1
                print(
                    f"[{split}] {particle}: {particle_index:,}/{events_per_class:,}",
                    flush=True,
                )
            if particle_index != events_per_class:
                raise RuntimeError(
                    f"{split}/{particle}: wrote {particle_index:,}, "
                    f"expected {events_per_class:,}"
                )
    finally:
        for array in arrays.values():
            array.flush()

    labels = np.asarray(arrays["labels"])
    if not np.all(labels[0::2] == 0) or not np.all(labels[1::2] == 1):
        raise RuntimeError(f"{split}: interleaved label audit failed")
    if max_energy_decode_relative_error > 5e-4:
        raise RuntimeError(
            f"{split}: energy decoding error too large: "
            f"{max_energy_decode_relative_error:.8g}"
        )

    summary = {
        "split": split,
        "events": 2 * events_per_class,
        "events_per_class": events_per_class,
        "label_counts": {"0": events_per_class, "1": events_per_class},
        "interleaving": "antiP then antiD for each selected index",
        "voxel_shape": [10, grid_x, grid_y],
        "tof_primary_features": 11,
        "max_energy_decode_relative_error": max_energy_decode_relative_error,
        "complete": True,
    }
    (split_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.grid_x < 1 or args.grid_y < 1:
        raise ValueError("grid dimensions must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (args.cache_dir / "_SUCCESS").is_file():
        raise RuntimeError(f"incomplete graph cache: {args.cache_dir}")

    cache_manifest_path = args.cache_dir / "cache_manifest.json"
    if not cache_manifest_path.is_file():
        raise FileNotFoundError(cache_manifest_path)
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    mean, std, normalizer = load_normalizer(args.cache_dir)
    args.output_dir.mkdir(parents=True)
    summaries = {
        split: export_split(
            args.cache_dir,
            args.output_dir,
            split,
            args.split_suffix,
            args.grid_x,
            args.grid_y,
            mean,
            std,
        )
        for split in SPLITS
    }
    for split, summary in summaries.items():
        source_summary = cache_manifest.get("splits", {}).get(split)
        if source_summary is None:
            raise RuntimeError(f"source cache manifest has no {split!r} split")
        if int(source_summary["events"]) != summary["events"]:
            raise RuntimeError(f"{split}: source/export event count mismatch")
        source_counts = {
            str(key): int(value)
            for key, value in source_summary["label_counts"].items()
        }
        if source_counts != summary["label_counts"]:
            raise RuntimeError(f"{split}: source/export label count mismatch")
    manifest = {
        "purpose": "CNN+DNN baseline on the exact strict-track-stop TreeRec graph sample",
        "source_cache": str(args.cache_dir.resolve()),
        "source": "TreeRec hitseries stored in the graph cache",
        "event_selection": cache_manifest.get("selection", "inherited from source cache"),
        "split_membership": "inherited unchanged from source cache",
        "input": {
            "voxel": [10, args.grid_x, args.grid_y],
            "voxel_value": "raw Si(Li) hit-energy sum",
            "tof_primary": 11,
            "beta_is_model_input": False,
            "mc_truth_is_model_input": False,
        },
        "energy_decoding": {
            "source_column": "graph.x[:, 3]",
            "operation": "expm1(x * std + mean)",
            "normalizer": normalizer,
        },
        "split_suffix": args.split_suffix,
        "source_cache_splits": cache_manifest.get("splits"),
        "splits": summaries,
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"complete: {args.output_dir}", flush=True)
    print("STRICT-TRACK TREERec CNN+DNN FIXED-GRID DATASET: VALID", flush=True)


if __name__ == "__main__":
    main()

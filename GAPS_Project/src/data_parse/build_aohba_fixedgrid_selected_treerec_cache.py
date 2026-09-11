#!/usr/bin/env python3
"""Build a TreeRec cache for the exact events in a fixed-grid export."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from GAPS_Project.src.data_parse.graph_builder import GraphBuilder


PARTICLES = {
    "antip": {"directory": "antiP", "label": 0, "pdg": -2212},
    "antid": {"directory": "antiD", "label": 1, "pdg": -1000010020},
}
SPLITS = {
    "train": (0, 80_000),
    "val": (80_000, 90_000),
    "test": (90_000, 100_000),
}
FEATURE_NAMES = (
    "x_mm",
    "y_mm",
    "z_mm",
    "log1p_energy",
    "log1p_time",
    "log1p_dedx",
)
HIT_BRANCHES = {
    "volume": "Rec/hitseries_/hitseries_.volume_id_",
    "energy": "Rec/hitseries_/hitseries_.energydep_",
    "position": "Rec/hitseries_/hitseries_.hit_position_",
    "time": "Rec/hitseries_/hitseries_.hit_time_",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixedgrid-raw-dir", type=Path, required=True)
    parser.add_argument("--fixedgrid-dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=8)
    return parser.parse_args()


def positions_for_event(value) -> np.ndarray:
    if len(value) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(
        [
            np.asarray(value["fX"], dtype=np.float32),
            np.asarray(value["fY"], dtype=np.float32),
            np.asarray(value["fZ"], dtype=np.float32),
        ],
        axis=1,
    )


def load_provenance(raw_dir: Path, particle: str) -> dict:
    directory = raw_dir / PARTICLES[particle]["directory"]
    if not (directory / "_SUCCESS").is_file():
        raise RuntimeError(f"incomplete fixed-grid export: {directory}")
    export_manifest = json.loads(
        (directory / "export_manifest.json").read_text()
    )
    file_indices = np.load(
        directory / "source_file_indices.npy", mmap_mode="r"
    )
    entries = np.load(directory / "source_entries.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy", mmap_mode="r")
    files = [
        Path(line)
        for line in (directory / "source_files.txt").read_text().splitlines()
        if line.strip()
    ]
    if len(file_indices) < 100_000 or len(entries) < 100_000:
        raise RuntimeError(f"{particle}: fewer than 100,000 selected events")
    if not np.all(labels[:100_000] == PARTICLES[particle]["label"]):
        raise RuntimeError(f"{particle}: label mismatch in fixed-grid export")
    if np.any(file_indices[:100_000] < 0) or np.any(
        file_indices[:100_000] >= len(files)
    ):
        raise RuntimeError(f"{particle}: invalid source file index")
    if np.any(entries[:100_000] < 0):
        raise RuntimeError(f"{particle}: invalid source ROOT entry")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source ROOT files: {missing[:3]}")
    return {
        "file_indices": file_indices,
        "entries": entries,
        "files": files,
        "selection": export_manifest.get(
            "selection", "stopped-toptrigger"
        ),
    }


def audit_assembled_dataset(
    dataset_dir: Path, provenance: dict[str, dict]
) -> None:
    for split, (start, stop) in SPLITS.items():
        directory = dataset_dir / f"{split}_nakagami_style_4M"
        file_indices = np.load(directory / "source_file_indices.npy")
        entries = np.load(directory / "source_entries.npy")
        particles = np.load(directory / "source_particle.npy")
        expected_count = 2 * (stop - start)
        if len(entries) != expected_count:
            raise RuntimeError(
                f"{split}: assembled event count {len(entries)} != {expected_count}"
            )
        checks = (
            np.array_equal(
                file_indices[0::2],
                provenance["antip"]["file_indices"][start:stop],
            ),
            np.array_equal(
                file_indices[1::2],
                provenance["antid"]["file_indices"][start:stop],
            ),
            np.array_equal(
                entries[0::2], provenance["antip"]["entries"][start:stop]
            ),
            np.array_equal(
                entries[1::2], provenance["antid"]["entries"][start:stop]
            ),
            np.all(particles[0::2] == 0),
            np.all(particles[1::2] == 1),
        )
        if not all(checks):
            raise RuntimeError(f"{split}: assembled provenance mismatch")
        print(f"[PAIR AUDIT] {split}: {expected_count:,} exact events")


def read_event_chunk(
    provenance: dict,
    particle: str,
    selected_start: int,
    selected_stop: int,
) -> list[dict]:
    selected_file_indices = np.asarray(
        provenance["file_indices"][selected_start:selected_stop]
    )
    selected_entries = np.asarray(
        provenance["entries"][selected_start:selected_stop]
    )
    events: list[dict | None] = [None] * len(selected_entries)

    for file_index in np.unique(selected_file_indices):
        output_indices = np.flatnonzero(selected_file_indices == file_index)
        root_entries = selected_entries[output_indices]
        entry_start = int(root_entries.min())
        entry_stop = int(root_entries.max()) + 1
        path = provenance["files"][int(file_index)]

        with uproot.open(path) as root_file:
            mc = root_file["TreeMc"]
            rec = root_file["TreeRec"]
            if mc.num_entries != rec.num_entries:
                raise RuntimeError(f"{path}: TreeMc/TreeRec entry mismatch")
            if entry_stop > rec.num_entries:
                raise RuntimeError(f"{path}: selected entry outside TreeRec")
            pdgs = np.asarray(
                mc["Mc/primaryPdg_"].array(
                    entry_start=entry_start,
                    entry_stop=entry_stop,
                    library="np",
                )
            ).reshape(-1)
            betas = np.asarray(
                mc["Mc/CEventBase/primaryBetaGenerated_"].array(
                    entry_start=entry_start,
                    entry_stop=entry_stop,
                    library="np",
                )
            ).reshape(-1)
            volume = rec[HIT_BRANCHES["volume"]].array(
                entry_start=entry_start, entry_stop=entry_stop, library="ak"
            )
            energy = rec[HIT_BRANCHES["energy"]].array(
                entry_start=entry_start, entry_stop=entry_stop, library="ak"
            )
            position = rec[HIT_BRANCHES["position"]].array(
                entry_start=entry_start, entry_stop=entry_stop, library="ak"
            )
            times = rec[HIT_BRANCHES["time"]].array(
                entry_start=entry_start, entry_stop=entry_stop, library="ak"
            )

        for output_index, root_entry in zip(output_indices, root_entries):
            local = int(root_entry) - entry_start
            event_volume = np.asarray(volume[local], dtype=np.int64)
            event_energy = np.asarray(energy[local], dtype=np.float32)
            event_position = positions_for_event(position[local])
            event_times = np.asarray(times[local], dtype=np.float32)
            lengths = {
                len(event_volume),
                len(event_energy),
                len(event_position),
                len(event_times),
            }
            if len(lengths) != 1:
                raise RuntimeError(f"{path}: hit-array mismatch at {root_entry}")
            if len(event_energy) <= 1:
                raise RuntimeError(
                    f"{path}: paired entry {root_entry} has N<=1 hits"
                )
            if int(pdgs[local]) != PARTICLES[particle]["pdg"]:
                raise RuntimeError(f"{path}: PDG mismatch at {root_entry}")
            if not np.isfinite(event_energy).all():
                raise RuntimeError(f"{path}: non-finite energy at {root_entry}")
            if not np.isfinite(event_position).all():
                raise RuntimeError(f"{path}: non-finite position at {root_entry}")
            if not np.isfinite(betas[local]):
                raise RuntimeError(f"{path}: non-finite beta at {root_entry}")
            events[int(output_index)] = {
                "energy": event_energy,
                "positions": event_position,
                "times": event_times,
                "volume_id": event_volume,
                "label": PARTICLES[particle]["pdg"],
                "beta": float(betas[local]),
                "source_file_index": int(file_index),
                "source_root_entry": int(root_entry),
                "source_selected_index": selected_start + int(output_index),
            }

    if any(event is None for event in events):
        raise RuntimeError(f"{particle}: failed to load every selected event")
    return list(events)


def iter_events(
    provenance: dict,
    particle: str,
    start: int,
    stop: int,
    chunk_size: int,
) -> Iterator[list[dict]]:
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        yield read_event_chunk(
            provenance, particle, chunk_start, chunk_stop
        )


def fit_normalizer(
    provenance: dict[str, dict], chunk_size: int, k: int
) -> tuple[np.ndarray, np.ndarray, int]:
    builder = GraphBuilder(k=k, normalize=False)
    sums = np.zeros(6, dtype=np.float64)
    sums_squared = np.zeros(6, dtype=np.float64)
    n_nodes = 0
    n_events = 0
    start, stop = SPLITS["train"]
    for particle in PARTICLES:
        for events in iter_events(
            provenance[particle], particle, start, stop, chunk_size
        ):
            for event in events:
                raw = builder.raw_node_features_from_dict(event)
                transformed = raw[:, :6].astype(np.float64, copy=True)
                transformed[:, 3:6] = np.log1p(
                    np.clip(transformed[:, 3:6], 0.0, None)
                )
                if not np.isfinite(transformed).all():
                    raise RuntimeError("non-finite transformed train feature")
                sums += transformed.sum(axis=0)
                sums_squared += np.square(transformed).sum(axis=0)
                n_nodes += len(transformed)
                n_events += 1
            print(
                f"[STATS] {particle}: {n_events:,} cumulative train events",
                flush=True,
            )
    if n_events != 160_000 or n_nodes == 0:
        raise RuntimeError(
            f"normalizer expected 160,000 events, found {n_events:,}"
        )
    mean = sums / n_nodes
    variance = np.maximum(sums_squared / n_nodes - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), n_nodes


def build_cache(
    args: argparse.Namespace,
    provenance: dict[str, dict],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    builder = GraphBuilder(
        k=args.k,
        normalize=True,
        normalization_mode="global_log",
        global_feature_mean=mean,
        global_feature_std=std,
    )
    summaries = {}
    for split, (start, stop) in SPLITS.items():
        label_counts = {0: 0, 1: 0}
        shard_count = 0
        for particle in PARTICLES:
            particle_shard = 0
            for events in iter_events(
                provenance[particle],
                particle,
                start,
                stop,
                args.chunk_size,
            ):
                graphs = []
                for event in events:
                    graph = builder.build_from_dict(event)
                    graph.source_file_index = torch.tensor(
                        [event["source_file_index"]], dtype=torch.long
                    )
                    graph.source_root_entry = torch.tensor(
                        [event["source_root_entry"]], dtype=torch.long
                    )
                    graph.source_selected_index = torch.tensor(
                        [event["source_selected_index"]], dtype=torch.long
                    )
                    if not torch.isfinite(graph.x).all():
                        raise RuntimeError("non-finite normalized graph feature")
                    graphs.append(graph)
                    label_counts[int(graph.y.item())] += 1

                destination = args.output_dir / (
                    f"{split}_{particle}_{particle_shard:03d}.pt"
                )
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(graphs, temporary)
                temporary.replace(destination)
                destination.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "split": split,
                            "particle": particle,
                            "selected_index_start": events[0][
                                "source_selected_index"
                            ],
                            "selected_index_stop": events[-1][
                                "source_selected_index"
                            ]
                            + 1,
                            "n_graphs": len(graphs),
                            "normalization": "train-only global_log",
                        },
                        indent=2,
                    )
                )
                print(f"[SAVE] {destination.name}: {len(graphs):,}")
                particle_shard += 1
                shard_count += 1
        expected_per_class = stop - start
        if label_counts != {0: expected_per_class, 1: expected_per_class}:
            raise RuntimeError(f"{split}: label counts {label_counts}")
        summaries[split] = {
            "events": 2 * expected_per_class,
            "events_per_class": expected_per_class,
            "label_counts": label_counts,
            "selected_range_per_class": [start, stop],
            "shards": shard_count,
        }
    return summaries


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1 or args.k < 1:
        raise ValueError("chunk size and k must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    provenance = {
        particle: load_provenance(args.fixedgrid_raw_dir, particle)
        for particle in PARTICLES
    }
    selections = {
        item["selection"] for item in provenance.values()
    }
    if len(selections) != 1:
        raise RuntimeError(
            f"particle provenance uses different selections: {selections}"
        )
    selection = selections.pop()
    if args.fixedgrid_dataset_dir is not None:
        audit_assembled_dataset(args.fixedgrid_dataset_dir, provenance)

    mean, std, train_nodes = fit_normalizer(
        provenance, args.chunk_size, args.k
    )
    normalizer = {
        "mode": "global_log",
        "fit_split": "train",
        "continuous_columns": [0, 1, 2, 3, 4, 5],
        "log1p_columns": [3, 4, 5],
        "unchanged_columns": {"6": "det_type", "7": "layer_norm"},
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_nodes": train_nodes,
        "train_events": 160_000,
    }
    (args.output_dir / "node_feature_normalizer.json").write_text(
        json.dumps(normalizer, indent=2)
    )
    print("normalizer:")
    for name, feature_mean, feature_std in zip(FEATURE_NAMES, mean, std):
        print(f"  {name:14s} mean={feature_mean:.7g} std={feature_std:.7g}")

    summaries = build_cache(args, provenance, mean, std)
    manifest = {
        "purpose": "TreeRec for exact Aohba TreeMc-selected events",
        "pairing": "source_file_indices.npy + source_entries.npy",
        "selection": selection,
        "fixedgrid_raw_dir": str(args.fixedgrid_raw_dir.resolve()),
        "fixedgrid_dataset_dir": (
            str(args.fixedgrid_dataset_dir.resolve())
            if args.fixedgrid_dataset_dir is not None
            else None
        ),
        "normalization": "global_log fitted on paired train events only",
        "k": args.k,
        "splits": summaries,
    }
    (args.output_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    (args.output_dir / "_SUCCESS").write_text("ok\n")
    print(json.dumps(summaries, indent=2))
    print(f"complete: {args.output_dir}")
    print("AOHBA FIXED-GRID-SELECTED TREEREC 200K CACHE: VALID")


if __name__ == "__main__":
    main()

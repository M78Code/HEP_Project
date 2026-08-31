#!/usr/bin/env python3
"""Build a source-disjoint old-domain TreeRec cache.

The training and validation ROOT files are reconstructed selections from
independent 2021 simulation sources.  The test ROOT files are the fixed paired
validation-source sample used for the TreeMc-versus-TreeRec audit.  Node
global-log normalization is fitted exclusively on training TreeRec hits.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from GAPS_Project.src.data_parse.graph_builder import GraphBuilder


PARTICLES = ("antip", "antid")
LABEL = {"antip": 0, "antid": 1}
PDG = {"antip": -2212, "antid": -1000010020}
SOURCE_PATTERN = re.compile(r"_(\d{10})_selected")
FEATURE_NAMES = (
    "x_mm",
    "y_mm",
    "z_mm",
    "log1p_energy",
    "log1p_time",
    "log1p_dedx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-reco-dir", type=Path, required=True)
    parser.add_argument("--test-reco-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--expected-train", type=int, default=2000)
    parser.add_argument("--expected-val", type=int, default=1000)
    parser.add_argument("--expected-test", type=int, default=2000)
    parser.add_argument("--test-antip-source", type=int, default=1627528714)
    parser.add_argument("--test-antid-source", type=int, default=1627550286)
    return parser.parse_args()


def particle_from_path(path: Path) -> str:
    name = path.name.lower()
    matches = [particle for particle in PARTICLES if particle in name]
    if len(matches) != 1:
        raise ValueError(f"cannot infer particle from {path}")
    return matches[0]


def source_from_path(path: Path) -> int:
    match = SOURCE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"cannot infer source ID from {path}")
    return int(match.group(1))


def split_paths(args: argparse.Namespace) -> dict[str, list[Path]]:
    paths = {
        "train": sorted(args.pilot_reco_dir.glob("reco_train_*.root")),
        "val": sorted(args.pilot_reco_dir.glob("reco_val_*.root")),
        "test": sorted(args.test_reco_dir.glob("reco_*.root")),
    }
    for split, split_files in paths.items():
        if not split_files:
            raise FileNotFoundError(f"no {split} ROOT files found")
    return paths


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


def scalar_array(tree, branch: str) -> np.ndarray:
    return np.asarray(tree[branch].array(library="np")).reshape(-1)


def load_root_events(path: Path, source_id: int, k: int) -> list[dict]:
    particle = particle_from_path(path)
    with uproot.open(path) as root_file:
        mc = root_file["TreeMc"]
        rec = root_file["TreeRec"]
        if mc.num_entries != rec.num_entries:
            raise RuntimeError(
                f"{path}: TreeMc/TreeRec mismatch "
                f"({mc.num_entries} != {rec.num_entries})"
            )
        pdgs = scalar_array(mc, "Mc/primaryPdg_")
        betas = scalar_array(mc, "Mc/CEventBase/primaryBetaGenerated_")
        volume = rec["Rec/hitseries_/hitseries_.volume_id_"].array(library="ak")
        energy = rec["Rec/hitseries_/hitseries_.energydep_"].array(library="ak")
        position = rec[
            "Rec/hitseries_/hitseries_.hit_position_"
        ].array(library="ak")
        times = rec["Rec/hitseries_/hitseries_.hit_time_"].array(library="ak")

    events = []
    for index in range(len(pdgs)):
        event_volume = np.asarray(volume[index], dtype=np.int64)
        event_energy = np.asarray(energy[index], dtype=np.float32)
        event_position = positions_for_event(position[index])
        event_times = np.asarray(times[index], dtype=np.float32)
        lengths = {
            len(event_volume),
            len(event_energy),
            len(event_position),
            len(event_times),
        }
        if len(lengths) != 1:
            raise RuntimeError(f"{path}: hit-array mismatch at entry {index}")
        if len(event_energy) <= k:
            raise RuntimeError(
                f"{path}: entry {index} has {len(event_energy)} hits, "
                f"not enough for k={k}"
            )
        if int(pdgs[index]) != PDG[particle]:
            raise RuntimeError(
                f"{path}: PDG mismatch at entry {index}: {int(pdgs[index])}"
            )
        if not np.isfinite(event_energy).all():
            raise RuntimeError(f"{path}: non-finite energy at entry {index}")
        if not np.isfinite(event_position).all():
            raise RuntimeError(f"{path}: non-finite position at entry {index}")
        if not np.isfinite(betas[index]):
            raise RuntimeError(f"{path}: non-finite MC beta at entry {index}")

        events.append(
            {
                "energy": event_energy,
                "positions": event_position,
                "times": event_times,
                "volume_id": event_volume,
                "label": PDG[particle],
                "beta": float(betas[index]),
                "particle": particle,
                "source_id": source_id,
                "root_entry": index,
            }
        )
    return events


def source_ids_by_split(
    args: argparse.Namespace,
    paths: dict[str, list[Path]],
) -> dict[str, set[int]]:
    sources = {
        "train": {source_from_path(path) for path in paths["train"]},
        "val": {source_from_path(path) for path in paths["val"]},
        "test": {args.test_antip_source, args.test_antid_source},
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sources[first] & sources[second]
        if overlap:
            raise RuntimeError(
                f"{first}/{second} source overlap: {sorted(overlap)}"
            )
    return sources


def fit_train_normalizer(
    paths: list[Path],
    k: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    builder = GraphBuilder(k=k, normalize=False)
    sums = np.zeros(6, dtype=np.float64)
    sums_squared = np.zeros(6, dtype=np.float64)
    n_nodes = 0
    n_events = 0

    for path in paths:
        source_id = source_from_path(path)
        events = load_root_events(path, source_id, k)
        for event in events:
            raw = builder.raw_node_features_from_dict(event)
            transformed = raw[:, :6].astype(np.float64, copy=True)
            transformed[:, 3:6] = np.log1p(
                np.clip(transformed[:, 3:6], 0.0, None)
            )
            if not np.isfinite(transformed).all():
                raise RuntimeError(f"non-finite train feature from {path}")
            sums += transformed.sum(axis=0)
            sums_squared += np.square(transformed).sum(axis=0)
            n_nodes += len(transformed)
            n_events += 1
        print(f"[train stats] {path.name}: {len(events):,} events", flush=True)

    if n_nodes == 0:
        raise RuntimeError("no training nodes")
    mean = sums / n_nodes
    variance = np.maximum(sums_squared / n_nodes - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), n_nodes, n_events


def test_source_id(args: argparse.Namespace, path: Path) -> int:
    particle = particle_from_path(path)
    return (
        args.test_antip_source
        if particle == "antip"
        else args.test_antid_source
    )


def build_split(
    args: argparse.Namespace,
    split: str,
    paths: list[Path],
    builder: GraphBuilder,
) -> dict:
    total = 0
    label_counts: Counter[int] = Counter()
    source_counts: Counter[int] = Counter()

    for shard_index, path in enumerate(paths):
        source_id = (
            test_source_id(args, path)
            if split == "test"
            else source_from_path(path)
        )
        events = load_root_events(path, source_id, args.k)
        graphs = []
        for event in events:
            graph = builder.build_from_dict(event)
            graph.source_id = torch.tensor([source_id], dtype=torch.long)
            graph.source_root_entry = torch.tensor(
                [event["root_entry"]], dtype=torch.long
            )
            graph.source_root_shard = torch.tensor(
                [shard_index], dtype=torch.long
            )
            if not torch.isfinite(graph.x).all():
                raise RuntimeError(f"non-finite graph.x produced from {path}")
            graphs.append(graph)
            label_counts[int(graph.y.item())] += 1
            source_counts[source_id] += 1

        particle = particle_from_path(path)
        destination = args.output_dir / (
            f"{split}_{particle}_{source_id}_{shard_index:02d}.pt"
        )
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(graphs, temporary)
        temporary.replace(destination)
        destination.with_suffix(".json").write_text(
            json.dumps(
                {
                    "split": split,
                    "particle": particle,
                    "source_id": source_id,
                    "source_root": str(path.resolve()),
                    "n_graphs": len(graphs),
                    "normalization": "train-only global_log",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        total += len(graphs)
        print(f"[{split}] saved {destination.name}: {len(graphs):,}", flush=True)

    return {
        "events": total,
        "label_counts": dict(sorted(label_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "shards": len(paths),
    }


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = split_paths(args)
    sources = source_ids_by_split(args, paths)
    print("source IDs:", {key: sorted(value) for key, value in sources.items()})

    mean, std, n_nodes, n_train_events = fit_train_normalizer(
        paths["train"], args.k
    )
    print("train node statistics:")
    for name, feature_mean, feature_std in zip(FEATURE_NAMES, mean, std):
        print(f"  {name:14s} mean={feature_mean:.7g} std={feature_std:.7g}")

    normalizer = {
        "mode": "global_log",
        "fit_split": "train",
        "continuous_columns": [0, 1, 2, 3, 4, 5],
        "log1p_columns": [3, 4, 5],
        "unchanged_columns": {"6": "det_type", "7": "layer_norm"},
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_nodes": n_nodes,
        "train_events": n_train_events,
        "train_sources": sorted(sources["train"]),
    }
    (args.output_dir / "node_feature_normalizer.json").write_text(
        json.dumps(normalizer, indent=2), encoding="utf-8"
    )

    builder = GraphBuilder(
        k=args.k,
        normalize=True,
        normalization_mode="global_log",
        global_feature_mean=mean,
        global_feature_std=std,
    )
    summaries = {
        split: build_split(args, split, paths[split], builder)
        for split in ("train", "val", "test")
    }
    expected = {
        "train": args.expected_train,
        "val": args.expected_val,
        "test": args.expected_test,
    }
    for split, expected_events in expected.items():
        if summaries[split]["events"] != expected_events:
            raise RuntimeError(
                f"{split}: expected {expected_events} events, "
                f"found {summaries[split]['events']}"
            )
        expected_per_class = expected_events // 2
        if summaries[split]["label_counts"] != {
            0: expected_per_class,
            1: expected_per_class,
        }:
            raise RuntimeError(
                f"{split}: unexpected label counts "
                f"{summaries[split]['label_counts']}"
            )

    manifest = {
        "purpose": "old-domain source-disjoint TreeRec pilot",
        "normalization": "global_log fitted on train only",
        "k": args.k,
        "pilot_reco_dir": str(args.pilot_reco_dir.resolve()),
        "test_reco_dir": str(args.test_reco_dir.resolve()),
        "sources": {key: sorted(value) for key, value in sources.items()},
        "splits": summaries,
    }
    (args.output_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"complete: {args.output_dir}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

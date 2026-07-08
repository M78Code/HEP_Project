#!/usr/bin/env python3
"""Export local430 GAPS pkl splits as Nakagami-old-style PyG graphs.

The old Nakagami CSV VolID graph uses node features close to:
  [log1p(edep), tof/50, stop_or_layer/10, volume_id/1e8] + pos

The local430 pkl files do not contain Nakagami's stoplayer column, so this
script uses a detector-layer-like value derived from volume_id as the third
node feature. This keeps the representation close in shape while avoiding use
of unavailable old-simulation metadata.

Important time-feature note:
  The original `tof/50` scaling was chosen for Nakagami's old CSV, where the
  TOF-like column is typically O(10). local430 `times` are not on the same
  numerical scale and can contain very large values, so directly using
  `time/50` can produce features of O(1e5-1e6) and destabilize AMP training.
  Here we use log compression instead:
      time_feature = log1p(max(time, 0)) / 20
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


NO_ANTIPROTON = -2212
NO_ANTIDEUTERON = -1000010020


def make_time_feature(times: np.ndarray) -> np.ndarray:
    """Return a bounded local430 time feature.

    Old Nakagami VolID CSV used a TOF-like value with a small range, so the
    previous `time / 50` scaling was only a rough attempt to match that scale.
    local430 times can be orders of magnitude larger; log compression keeps
    the ordering information without letting rare extreme values dominate.
    """
    finite = np.nan_to_num(times, nan=0.0, posinf=0.0, neginf=0.0)
    non_negative = np.clip(finite, 0.0, None)
    return (np.log1p(non_negative) / 20.0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path, help="directory containing train/val/test.pkl")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-per-class-per-split",
        type=int,
        default=0,
        help="balanced cap per class in each split; 0 keeps all available minority-class events",
    )
    p.add_argument(
        "--third-feature",
        choices=["layer", "zero"],
        default="layer",
        help="replacement for old CSV stoplayer feature. 'layer' uses volume_id-derived layer; 'zero' ablates it.",
    )
    return p.parse_args()


def load_events(path: Path) -> list[dict]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "events" in obj:
        return obj["events"]
    if isinstance(obj, list):
        return obj
    raise TypeError(f"unsupported pkl structure in {path}: {type(obj)}")


def label_to_binary(label: int) -> int | None:
    if int(label) == NO_ANTIPROTON:
        return 0
    if int(label) == NO_ANTIDEUTERON:
        return 1
    return None


def graph_from_event(event: dict, knn_k: int, third_feature: str) -> Data | None:
    energies = np.asarray(event["energy"], dtype=np.float32)
    positions = np.asarray(event["positions"], dtype=np.float32)
    times = np.asarray(event["times"], dtype=np.float32)
    volume_ids = np.asarray(event["volume_id"], dtype=np.int64)
    y = label_to_binary(int(event["label"]))
    if y is None or len(energies) < 2:
        return None

    pos = torch.tensor(positions / 1000.0, dtype=torch.float32)
    edep = torch.tensor(np.log1p(np.clip(energies, 0.0, None)), dtype=torch.float32).view(-1, 1)
    tof = torch.tensor(make_time_feature(times), dtype=torch.float32).view(-1, 1)
    vol = torch.tensor(volume_ids, dtype=torch.long)

    if third_feature == "layer":
        layer_idx = (volume_ids // 1_000_000).astype(np.float32)
        layer_like = ((layer_idx % 100.0) / 10.0).astype(np.float32)
    else:
        layer_like = np.zeros_like(energies, dtype=np.float32)

    x = torch.cat(
        [
            edep,
            tof,
            torch.tensor(layer_like, dtype=torch.float32).view(-1, 1),
            vol.to(torch.float32).view(-1, 1) / 1.0e8,
        ],
        dim=1,
    )

    k_eff = min(knn_k, pos.size(0) - 1)
    edge_index = knn_graph(pos, k=k_eff, loop=False)
    beta = float(event.get("beta", 0.0))

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        volume_id=vol,
        y=torch.tensor(y, dtype=torch.long),
        beta=torch.tensor(beta, dtype=torch.float32),
    )


def balanced_select(graphs: list[Data], max_per_class: int, seed: int) -> list[Data]:
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for graph in graphs:
        by_label[int(graph.y)].append(graph)

    n = min(len(by_label[0]), len(by_label[1]))
    if max_per_class > 0:
        n = min(n, max_per_class)

    selected = []
    for label in [0, 1]:
        xs = by_label[label]
        rng.shuffle(xs)
        selected.extend(xs[:n])
    rng.shuffle(selected)
    return selected


def export_split(args: argparse.Namespace, split: str) -> dict[str, object]:
    path = args.input_dir / f"{split}.pkl"
    events = load_events(path)
    graphs = []
    skipped = 0
    for event in events:
        graph = graph_from_event(event, args.knn_k, args.third_feature)
        if graph is None:
            skipped += 1
            continue
        graphs.append(graph)

    selected = balanced_select(graphs, args.max_per_class_per_split, args.seed)
    labels = torch.tensor([int(g.y) for g in selected], dtype=torch.long)
    out_path = args.output_dir / f"{split}.pt"
    torch.save(selected, out_path)

    counts = torch.bincount(labels, minlength=2).tolist()
    print(f"{split}: raw_events={len(events):,} graphs={len(graphs):,} selected={len(selected):,} labels={counts}")
    print(f"saved {out_path}")
    return {
        "split": split,
        "raw_events": len(events),
        "graphs": len(graphs),
        "selected": len(selected),
        "label_counts": counts,
        "skipped": skipped,
        "output": str(out_path),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_dir": str(args.input_dir),
        "knn_k": args.knn_k,
        "max_per_class_per_split": args.max_per_class_per_split,
        "third_feature": args.third_feature,
        "time_feature": "log1p(max(time,0))/20",
        "splits": [],
    }
    for split in ["train", "val", "test"]:
        summary["splits"].append(export_split(args, split))

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("done:", args.output_dir)


if __name__ == "__main__":
    main()

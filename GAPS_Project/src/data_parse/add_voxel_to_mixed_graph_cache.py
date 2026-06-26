#!/usr/bin/env python
"""Add Si(Li) voxel tensors to mixed PyG graph-cache shards.

This script is intended for the Aohba 4M mixed graph cache.  The mixed
cache already stores random_seed and source_event_index for each graph,
so the original preprocessed pkl event can be recovered and converted to
the same Si(Li) voxel representation used by FusedGravNet.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import torch
from tqdm import tqdm

from GAPS_Project.src.data_parse.voxelizer import build_sili_voxel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *_mixed_*.pt graph-cache shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for shards with graph.voxel attached.",
    )
    parser.add_argument(
        "--pkl-base-dir",
        type=Path,
        default=Path("/mnt/ynakagami3/aohba_preprocess"),
        help="Base directory containing antiD/ and antiP/ preprocessed pkl files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--grid-x", type=int, default=20)
    parser.add_argument("--grid-y", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Reload each output shard after writing and check voxel presence.",
    )
    return parser.parse_args()


def is_complete(pt_path: Path, json_path: Path) -> bool:
    if not pt_path.exists() or not json_path.exists():
        return False
    try:
        with json_path.open() as f:
            row = json.load(f)
        return bool(row.get("complete")) and int(row.get("n_graphs", -1)) > 0
    except Exception:
        return False


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_events(pkl_path: Path):
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    return obj["events"] if isinstance(obj, dict) else obj


def add_voxels_to_shard(
    input_path: Path,
    output_path: Path,
    pkl_base_dir: Path,
    grid_x: int,
    grid_y: int,
) -> dict:
    started = time.time()
    graphs = torch.load(input_path, map_location="cpu", weights_only=False)
    loaded_events = {}
    label_counts = {"0": 0, "1": 0}
    voxel_sum = 0.0
    nonzero_voxels = 0

    for graph in tqdm(graphs, desc=input_path.name, leave=False):
        label = int(graph.y.item())
        particle = "antiD" if label == 1 else "antiP"
        seed = int(graph.random_seed.item())
        source_event_index = int(graph.source_event_index.item())
        pkl_path = pkl_base_dir / particle / f"{particle}_2tof_FTFP_BERT_{seed}.pkl"

        if pkl_path not in loaded_events:
            if not pkl_path.exists():
                raise FileNotFoundError(f"missing source pkl: {pkl_path}")
            loaded_events[pkl_path] = load_events(pkl_path)

        event = loaded_events[pkl_path][source_event_index]
        voxel = build_sili_voxel(event, grid_x=grid_x, grid_y=grid_y)
        voxel_tensor = torch.as_tensor(voxel, dtype=torch.float32)
        graph.voxel = voxel_tensor

        label_counts[str(label)] += 1
        voxel_sum += float(voxel_tensor.sum().item())
        nonzero_voxels += int(torch.count_nonzero(voxel_tensor).item())

    atomic_torch_save(graphs, output_path)

    del graphs
    loaded_events.clear()
    gc.collect()

    return {
        "input": str(input_path),
        "output": str(output_path),
        "n_graphs": sum(label_counts.values()),
        "label_counts": label_counts,
        "grid": [10, grid_x, grid_y],
        "voxel_sum": voxel_sum,
        "nonzero_voxels": nonzero_voxels,
        "elapsed_sec": round(time.time() - started, 2),
        "complete": True,
    }


def verify_output(path: Path) -> None:
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    if not graphs:
        raise RuntimeError(f"empty shard: {path}")
    graph = graphs[0]
    if not hasattr(graph, "voxel"):
        raise RuntimeError(f"voxel missing in {path}")
    if tuple(graph.voxel.shape)[:1] != (10,):
        raise RuntimeError(f"unexpected voxel shape in {path}: {tuple(graph.voxel.shape)}")
    if not torch.isfinite(graph.voxel).all().item():
        raise RuntimeError(f"non-finite voxel values in {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split in args.splits:
        input_files = sorted(args.input_dir.glob(f"{split}_mixed_*.pt"))
        if not input_files:
            raise FileNotFoundError(f"no input shards found for split={split}: {args.input_dir}")

        print(f"\n=== {split}: {len(input_files)} shards ===", flush=True)
        for input_path in input_files:
            output_path = args.output_dir / input_path.name
            json_path = output_path.with_suffix(".json")

            if not args.overwrite and is_complete(output_path, json_path):
                print(f"skip complete: {output_path.name}", flush=True)
                with json_path.open() as f:
                    all_rows.append(json.load(f))
                continue

            row = add_voxels_to_shard(
                input_path=input_path,
                output_path=output_path,
                pkl_base_dir=args.pkl_base_dir,
                grid_x=args.grid_x,
                grid_y=args.grid_y,
            )
            with json_path.open("w") as f:
                json.dump(row, f, indent=2)

            if args.verify:
                verify_output(output_path)

            all_rows.append(row)
            print(
                f"saved {output_path.name}: {row['n_graphs']:,} graphs, "
                f"labels={row['label_counts']}, elapsed={row['elapsed_sec']}s",
                flush=True,
            )

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "pkl_base_dir": str(args.pkl_base_dir),
        "splits": args.splits,
        "grid": [10, args.grid_x, args.grid_y],
        "n_shards": len(all_rows),
        "n_graphs": sum(row["n_graphs"] for row in all_rows),
        "label_counts": {
            "0": sum(row["label_counts"].get("0", 0) for row in all_rows),
            "1": sum(row["label_counts"].get("1", 0) for row in all_rows),
        },
        "rows": all_rows,
    }
    with (args.output_dir / "summary_all.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== complete ==========", flush=True)
    print(f"graphs: {summary['n_graphs']:,}", flush=True)
    print(f"labels: {summary['label_counts']}", flush=True)
    print(f"output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Export Aohba mixed graph-cache shards to Nakagami-style three-input arrays.

The output format matches ``nakagami/data_parse/three_input_dataset.py``:

  - voxels.npy       : Si(Li) voxel, (N, 10, 12, 12) by default
  - tof_paddles.npy  : TOF paddle energy, (N, 172)
  - tof_primary.npy  : 11-D TOF features from TreeRec, (N, 11)
  - labels.npy       : antiP=0, antiD=1

By default voxels are exported as raw energy sums, matching the Nakagami-style
three-input interface more closely than the log1p preprocessing used by some
newer baselines.  Use ``--log1p-voxel`` only when a deliberately log-scaled
comparison is needed.

This script does not create an exact onlyPrimary reproduction.  It keeps the
Nakagami-style input shape and three-input model interface, while the values
are derived from the new Aohba TreeRec graph-cache/pkl data.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from GAPS_Project.src.data_parse.voxelizer import build_sili_voxel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing train/val/test *_mixed_*.pt graph-cache shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Nakagami-style three-input .npy arrays.",
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
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    parser.add_argument("--grid-x", type=int, default=12)
    parser.add_argument("--grid-y", type=int, default=12)
    parser.add_argument(
        "--log1p-voxel",
        action="store_true",
        help="Store log1p(voxel). Default stores raw energy-sum voxels.",
    )
    parser.add_argument(
        "--output-suffix",
        default="nakagami_style_4M",
        help="Split output directory suffix, e.g. train_<suffix>.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Reload exported arrays and check shape/label counts after each split.",
    )
    return parser.parse_args()


def count_graphs(pt_path: Path) -> int:
    json_path = pt_path.with_suffix(".json")
    if json_path.exists():
        with json_path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        if "n_graphs" in row:
            return int(row["n_graphs"])
    return len(torch.load(pt_path, map_location="cpu", weights_only=False))


def load_events(pkl_path: Path):
    with pkl_path.open("rb") as handle:
        obj = pickle.load(handle)
    return obj["events"] if isinstance(obj, dict) else obj


def split_complete(split_dir: Path, n_events: int, grid_x: int, grid_y: int) -> bool:
    summary_path = split_dir / "summary.json"
    required = ["voxels.npy", "tof_paddles.npy", "tof_primary.npy", "labels.npy", "betas.npy"]
    if not summary_path.exists() or not all((split_dir / name).exists() for name in required):
        return False
    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        return (
            bool(summary.get("complete"))
            and int(summary.get("n_events", -1)) == n_events
            and list(summary.get("voxel_shape", [])) == [10, grid_x, grid_y]
        )
    except Exception:
        return False


def allocate_arrays(split_dir: Path, n_events: int, grid_x: int, grid_y: int):
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "voxels": np.lib.format.open_memmap(
            split_dir / "voxels.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_events, 10, grid_x, grid_y),
        ),
        "tof_paddles": np.lib.format.open_memmap(
            split_dir / "tof_paddles.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_events, 172),
        ),
        "tof_primary": np.lib.format.open_memmap(
            split_dir / "tof_primary.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_events, 11),
        ),
        "labels": np.lib.format.open_memmap(
            split_dir / "labels.npy",
            mode="w+",
            dtype=np.int64,
            shape=(n_events,),
        ),
        "betas": np.lib.format.open_memmap(
            split_dir / "betas.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_events,),
        ),
    }


def flush_arrays(arrays) -> None:
    for array in arrays.values():
        array.flush()


def export_split(
    split: str,
    input_files: list[Path],
    output_dir: Path,
    output_suffix: str,
    pkl_base_dir: Path,
    grid_x: int,
    grid_y: int,
    log1p_voxel: bool,
    overwrite: bool,
    verify: bool,
) -> dict:
    split_dir = output_dir / f"{split}_{output_suffix}"
    n_events = sum(count_graphs(path) for path in input_files)

    if not overwrite and split_complete(split_dir, n_events, grid_x, grid_y):
        print(f"skip complete: {split_dir}", flush=True)
        with (split_dir / "summary.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    arrays = allocate_arrays(split_dir, n_events, grid_x, grid_y)
    loaded_events = {}
    label_counts = {"0": 0, "1": 0}
    voxel_sum = 0.0
    nonzero_voxels = 0
    cursor = 0
    started = time.time()

    try:
        for pt_path in input_files:
            graphs = torch.load(pt_path, map_location="cpu", weights_only=False)
            for graph in tqdm(graphs, desc=f"{split}:{pt_path.name}", leave=False, dynamic_ncols=True):
                if not hasattr(graph, "tof_paddle_energy"):
                    raise RuntimeError(f"tof_paddle_energy missing in {pt_path}")
                if not hasattr(graph, "tof_feat"):
                    raise RuntimeError(f"tof_feat missing in {pt_path}")

                label = int(graph.y.view(()).item())
                particle = "antiD" if label == 1 else "antiP"
                seed = int(graph.random_seed.view(()).item())
                source_event_index = int(graph.source_event_index.view(()).item())
                pkl_path = pkl_base_dir / particle / f"{particle}_2tof_FTFP_BERT_{seed}.pkl"

                if pkl_path not in loaded_events:
                    if not pkl_path.exists():
                        raise FileNotFoundError(f"missing source pkl: {pkl_path}")
                    loaded_events[pkl_path] = load_events(pkl_path)

                event = loaded_events[pkl_path][source_event_index]
                voxel = build_sili_voxel(event, grid_x=grid_x, grid_y=grid_y)
                if log1p_voxel:
                    voxel = np.log1p(voxel)

                tof_paddle = graph.tof_paddle_energy.detach().cpu().numpy().astype(np.float32, copy=False)
                tof_primary = graph.tof_feat.detach().cpu().numpy().astype(np.float32, copy=False)
                if tof_paddle.shape != (172,):
                    raise RuntimeError(f"unexpected tof_paddle shape in {pt_path}: {tof_paddle.shape}")
                if tof_primary.shape != (11,):
                    raise RuntimeError(f"unexpected tof_feat shape in {pt_path}: {tof_primary.shape}")

                arrays["voxels"][cursor] = voxel
                arrays["tof_paddles"][cursor] = tof_paddle
                arrays["tof_primary"][cursor] = tof_primary
                arrays["labels"][cursor] = label
                arrays["betas"][cursor] = (
                    float(graph.mc_beta.view(()).item()) if hasattr(graph, "mc_beta") else 0.0
                )

                label_counts[str(label)] += 1
                voxel_sum += float(voxel.sum())
                nonzero_voxels += int(np.count_nonzero(voxel))
                cursor += 1

            del graphs
            loaded_events.clear()
            gc.collect()
    finally:
        flush_arrays(arrays)
        del arrays
        loaded_events.clear()
        gc.collect()

    if cursor != n_events:
        raise RuntimeError(f"{split}: exported {cursor} events, expected {n_events}")

    summary = {
        "split": split,
        "input_dir": str(input_files[0].parent),
        "output_dir": str(split_dir),
        "pkl_base_dir": str(pkl_base_dir),
        "n_shards": len(input_files),
        "n_events": n_events,
        "label_counts": label_counts,
        "voxel_shape": [10, grid_x, grid_y],
        "tof_paddle_dim": 172,
        "tof_primary_dim": 11,
        "voxel_transform": "log1p" if log1p_voxel else "raw",
        "voxel_sum": voxel_sum,
        "nonzero_voxels": nonzero_voxels,
        "elapsed_sec": round(time.time() - started, 2),
        "complete": True,
    }
    with (split_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if verify:
        voxels = np.load(split_dir / "voxels.npy", mmap_mode="r")
        tof_paddles = np.load(split_dir / "tof_paddles.npy", mmap_mode="r")
        tof_primary = np.load(split_dir / "tof_primary.npy", mmap_mode="r")
        labels = np.load(split_dir / "labels.npy", mmap_mode="r")
        if voxels.shape != (n_events, 10, grid_x, grid_y):
            raise RuntimeError(f"bad voxel shape: {voxels.shape}")
        if tof_paddles.shape != (n_events, 172):
            raise RuntimeError(f"bad tof_paddles shape: {tof_paddles.shape}")
        if tof_primary.shape != (n_events, 11):
            raise RuntimeError(f"bad tof_primary shape: {tof_primary.shape}")
        if labels.shape != (n_events,):
            raise RuntimeError(f"bad labels shape: {labels.shape}")

    print(
        f"saved {split_dir}: {n_events:,} events, labels={label_counts}, "
        f"elapsed={summary['elapsed_sec']}s",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split in args.splits:
        input_files = sorted(args.input_dir.glob(f"{split}_mixed_*.pt"))
        if not input_files:
            raise FileNotFoundError(f"no {split}_mixed_*.pt files in {args.input_dir}")
        print(f"\n=== {split}: {len(input_files)} shards ===", flush=True)
        summaries.append(
            export_split(
                split=split,
                input_files=input_files,
                output_dir=args.output_dir,
                output_suffix=args.output_suffix,
                pkl_base_dir=args.pkl_base_dir,
                grid_x=args.grid_x,
                grid_y=args.grid_y,
                log1p_voxel=args.log1p_voxel,
                overwrite=args.overwrite,
                verify=args.verify,
            )
        )

    all_summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "pkl_base_dir": str(args.pkl_base_dir),
        "splits": args.splits,
        "voxel_shape": [10, args.grid_x, args.grid_y],
        "voxel_transform": "log1p" if args.log1p_voxel else "raw",
        "n_events": sum(row["n_events"] for row in summaries),
        "label_counts": {
            "0": sum(row["label_counts"].get("0", 0) for row in summaries),
            "1": sum(row["label_counts"].get("1", 0) for row in summaries),
        },
        "summaries": summaries,
        "complete": True,
    }
    with (args.output_dir / "summary_all.json").open("w", encoding="utf-8") as handle:
        json.dump(all_summary, handle, indent=2, ensure_ascii=False)

    print("\n========== complete ==========", flush=True)
    print(f"events: {all_summary['n_events']:,}", flush=True)
    print(f"labels: {all_summary['label_counts']}", flush=True)
    print(f"output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inventory inputs needed to reproduce the legacy 4M CNN+DNN route.

This is a factual preflight only.  It verifies what the strict TreeRec cache
retains, what tensor inputs the historical checkpoint requires, and whether a
specified legacy dataset contains an explicit 9-D TOF-energy array plus event
identity fields for exact event pairing.  It does not train a model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


SPLITS = ("train", "val", "test")
MAX_LEGACY_FILES = 160


def shape_of(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return list(value.shape)
    return None


def graph_inventory(cache_dir: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for split in SPLITS:
        files = sorted(cache_dir.glob(f"{split}_*.pt"))
        if not files:
            raise FileNotFoundError(f"no {split}_*.pt shards in {cache_dir}")
        graphs = torch.load(files[0], map_location="cpu", weights_only=False)
        if not graphs:
            raise RuntimeError(f"empty shard: {files[0]}")
        graph = graphs[0]
        fields = {}
        for key in graph.keys():
            fields[key] = {
                "shape": shape_of(getattr(graph, key)),
                "dtype": str(getattr(graph, key).dtype)
                if isinstance(getattr(graph, key), torch.Tensor)
                else type(getattr(graph, key)).__name__,
            }
        inventory[split] = {
            "first_shard": str(files[0]),
            "shards": len(files),
            "fields": fields,
            "legacy_9d_tof_present": any(
                fields[name]["shape"] == [9]
                for name in fields
                if "tof" in name.lower()
            ),
        }
        del graphs
    return inventory


def npy_or_npz_summary(path: Path) -> dict[str, Any]:
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        return {"kind": "npy", "shape": list(array.shape), "dtype": str(array.dtype)}
    with np.load(path, mmap_mode="r") as archive:
        return {
            "kind": "npz",
            "arrays": {
                key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
                for key in archive.files
            },
        }


def legacy_inventory(legacy_dir: Path | None) -> dict[str, Any] | None:
    if legacy_dir is None:
        return None
    if not legacy_dir.is_dir():
        return {"path": str(legacy_dir), "exists": False, "files": []}
    paths = sorted(
        path
        for path in legacy_dir.rglob("*")
        if path.is_file() and path.suffix in {".npy", ".npz"}
    )
    records = []
    for path in paths[:MAX_LEGACY_FILES]:
        try:
            summary = npy_or_npz_summary(path)
        except Exception as exc:  # Audit should inventory corrupt artifacts too.
            summary = {"error": f"{type(exc).__name__}: {exc}"}
        records.append({"path": str(path), **summary})
    return {
        "path": str(legacy_dir),
        "exists": True,
        "files_found": len(paths),
        "files_reported": len(records),
        "files": records,
    }


def checkpoint_inventory(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"unexpected checkpoint format: {type(payload).__name__}")
    dnn_first = state.get("dnn.0.weight")
    return {
        "path": str(path),
        "top_level_type": type(payload).__name__,
        "state_tensors": len(state),
        "state_elements": int(sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor))),
        "dnn_first_weight_shape": shape_of(dnn_first),
        "requires_9d_tof": shape_of(dnn_first) == [256, 9],
        "has_three_residual_stages": all(
            any(key.startswith(prefix) for key in state)
            for prefix in ("res1.", "res2.", "res3.")
        ),
    }


def markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Legacy CNN+DNN input mapping inventory",
        "",
        f"Generated: {audit['generated_at']}",
        "",
        "This file is an inventory of available inputs and checkpoint requirements. It contains no performance interpretation.",
        "",
        "## Historical checkpoint",
        "",
    ]
    for key, value in audit["historical_checkpoint"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Strict TreeRec graph-cache fields", ""]
    for split, summary in audit["strict_cache"].items():
        lines += [f"### {split}", "", f"- first shard: `{summary['first_shard']}`", f"- legacy_9d_tof_present: `{summary['legacy_9d_tof_present']}`", ""]
        lines += ["| Field | Shape | Dtype |", "| --- | --- | --- |"]
        for name, details in sorted(summary["fields"].items()):
            lines.append(f"| {name} | `{details['shape']}` | `{details['dtype']}` |")
        lines.append("")

    legacy = audit["legacy_dataset"]
    lines += ["## Candidate legacy-input dataset", ""]
    if legacy is None:
        lines.append("No legacy dataset path was supplied.")
    elif not legacy["exists"]:
        lines.append(f"Path does not exist: `{legacy['path']}`")
    else:
        lines.append(f"- Path: `{legacy['path']}`")
        lines.append(f"- Arrays found/reported: `{legacy['files_found']}` / `{legacy['files_reported']}`")
        lines += ["", "| File | Array layout |", "| --- | --- |"]
        for item in legacy["files"]:
            detail = item.get("arrays", item.get("shape", item.get("error")))
            lines.append(f"| `{item['path']}` | `{detail}` |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-cache", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.strict_cache / "_SUCCESS").is_file():
        raise FileNotFoundError(f"incomplete strict cache: {args.strict_cache}")
    if not args.historical_checkpoint.is_file():
        raise FileNotFoundError(f"historical checkpoint not found: {args.historical_checkpoint}")
    audit = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "historical_checkpoint": checkpoint_inventory(args.historical_checkpoint),
        "strict_cache": graph_inventory(args.strict_cache),
        "legacy_dataset": legacy_inventory(args.legacy_dataset),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "legacy_cnndnn_input_inventory.json"
    markdown_path = args.out_dir / "legacy_cnndnn_input_inventory.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown(audit), encoding="utf-8")
    print(f"historical checkpoint: {args.historical_checkpoint}")
    print(f"strict cache         : {args.strict_cache}")
    print(f"legacy dataset       : {args.legacy_dataset or 'not supplied'}")
    print(f"output               : {args.out_dir}")
    print("LEGACY CNN+DNN INPUT INVENTORY: COMPLETE")


if __name__ == "__main__":
    main()

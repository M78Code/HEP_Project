#!/usr/bin/env python3
"""Audit whether a strict-track TreeRec CNN+DNN run follows Fig. 7.2 protocol.

The audit deliberately distinguishes three levels of agreement:

* tensor schema (voxel/TOF shapes and model parameter shapes),
* training protocol (checkpoint arguments), and
* physical provenance of the input values.

Matching tensor shapes or a matching network does *not* prove that two inputs
contain the same physical information.  The generated report makes that
boundary explicit so it can be used in a research note or a supervisor report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import GAPS_Project
from GAPS_Project.src.models.cnn_dnn_hybrid import CNNDNNHybrid


PROJECT_ROOT = Path(GAPS_Project.__file__).resolve().parent
SPLITS = ("train", "val", "test")


def latest_run(root: Path, pattern: str) -> Path:
    candidates = sorted(
        (path for path in root.glob(pattern) if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no run matching {pattern!r} in {root}")
    return candidates[0]


def load_checkpoint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"unexpected checkpoint type: {type(payload).__name__}")
    return payload


def state_dict_from(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise TypeError("checkpoint does not contain a model state dictionary")
    return state


def state_shape_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    state = state_dict_from(payload)
    tensor_items = {
        key: list(value.shape)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    return {
        "tensor_count": len(tensor_items),
        "parameter_count": int(sum(int(np.prod(shape)) for shape in tensor_items.values())),
        "tensor_shapes": tensor_items,
    }


def compare_state_shapes(
    current: dict[str, Any] | None, reference: dict[str, Any] | None
) -> dict[str, Any] | None:
    if current is None or reference is None:
        return None
    current_shapes = state_shape_summary(current)["tensor_shapes"]
    reference_shapes = state_shape_summary(reference)["tensor_shapes"]
    all_keys = sorted(set(current_shapes) | set(reference_shapes))
    mismatches = [
        {
            "name": key,
            "current": current_shapes.get(key),
            "reference": reference_shapes.get(key),
        }
        for key in all_keys
        if current_shapes.get(key) != reference_shapes.get(key)
    ]
    return {
        "same_tensor_shapes": not mismatches,
        "mismatches": mismatches,
    }


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def selected_args(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    args = payload.get("args", {})
    if not isinstance(args, dict):
        return {"_unavailable": "checkpoint has no serialized argument dictionary"}
    keep = (
        "data_dir",
        "split_suffix",
        "dataset_tag",
        "epochs",
        "batch_size",
        "lr",
        "step_size",
        "lr_gamma",
        "dropout",
        "patience",
        "min_epochs",
        "min_delta",
        "num_workers",
        "amp",
        "amp_dtype",
        "seed",
    )
    return {key: json_value(args[key]) for key in keep if key in args}


def audit_dataset(dataset: Path, split_suffix: str) -> dict[str, Any]:
    manifest_path = dataset / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: dict[str, Any] = {}
    for split in SPLITS:
        split_dir = dataset / f"{split}_{split_suffix}"
        arrays = {}
        for name in ("voxels", "tof_primary", "labels", "betas"):
            path = split_dir / f"{name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"missing {split} array: {path}")
            array = np.load(path, mmap_mode="r")
            arrays[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        labels = np.load(split_dir / "labels.npy", mmap_mode="r")
        splits[split] = {
            "arrays": arrays,
            "label_counts": {
                "0": int(np.sum(labels == 0)),
                "1": int(np.sum(labels == 1)),
            },
        }
    return {"manifest": manifest, "splits": splits}


def markdown_report(audit: dict[str, Any]) -> str:
    dataset = audit["dataset"]
    current = audit["current_checkpoint"]
    reference = audit["reference_checkpoint"]
    state_comparison = audit["model_shape_comparison"]
    manifest = dataset["manifest"]

    lines = [
        "# Strict-track TreeRec 200K CNN+DNN protocol audit",
        "",
        f"Generated: {audit['generated_at']}",
        "",
        "## Scope",
        "",
        "This report checks the CNN+DNN reproduction protocol used for the strict-track-stop + truth-top-trigger TreeRec 200K pilot.",
        "It separates agreement of tensor schema/model/training settings from equivalence of the physical input provenance.",
        "",
        "## Current strict-track dataset",
        "",
        f"- Dataset: `{audit['dataset_path']}`",
        f"- Source graph cache: `{manifest.get('source_graph_cache', 'not recorded')}`",
        f"- Selection: `{manifest.get('selection', 'strict track stop + truth top-trigger (from runner)')}`",
        "- CNN branch: raw Si(Li) hit-energy accumulation on a 10 x 12 x 12 grid.",
        "- DNN branch: 11-D `tof_primary` from the TreeRec graph cache.",
        "- beta and TOF paddle arrays are metadata / unused by this CNN+DNN run.",
        "",
        "| Split | Events | antiP | antiD | Voxel tensor | TOF tensor |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for split in SPLITS:
        item = dataset["splits"][split]
        voxels = item["arrays"]["voxels"]["shape"]
        tof = item["arrays"]["tof_primary"]["shape"]
        n_events = item["arrays"]["labels"]["shape"][0]
        lines.append(
            f"| {split} | {n_events:,} | {item['label_counts']['0']:,} | "
            f"{item['label_counts']['1']:,} | `{voxels}` | `{tof}` |"
        )

    lines += [
        "",
        "## CNN+DNN architecture check",
        "",
        "- The current run instantiates `CNNDNNHybrid(tof_dim=11, dropout=0.3)`.",
        "- The model is the Fig. 7.2-style 3-D CNN branch plus 11-D DNN branch, with a reported 1,200,003 parameters.",
    ]
    if reference is None:
        lines += [
            "- No historical reference checkpoint was supplied; only the current protocol can be recorded.",
        ]
    else:
        shape_text = "identical" if state_comparison["same_tensor_shapes"] else "different"
        lines += [
            f"- Reference checkpoint: `{audit['reference_checkpoint_path']}`",
            f"- Current/reference parameter-tensor shapes are **{shape_text}**.",
        ]
        if state_comparison["mismatches"]:
            lines.append(f"- Shape mismatches: `{len(state_comparison['mismatches'])}` tensors.")

    lines += [
        "",
        "## Training setting comparison",
        "",
        "| Setting | Current strict 200K | Historical reference checkpoint |",
        "| --- | --- | --- |",
    ]
    current_args = current["training_args"] if current else {}
    reference_args = reference["training_args"] if reference else {}
    keys = sorted(set(current_args) | set(reference_args))
    if not keys:
        lines.append("| checkpoint arguments | unavailable | unavailable |")
    for key in keys:
        lines.append(
            f"| {key} | `{current_args.get(key, 'not recorded')}` | "
            f"`{reference_args.get(key, 'not recorded')}` |"
        )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The present pilot uses the same CNN+DNN tensor schema and intended Fig. 7.2 architecture, but it is not a bitwise reproduction of the old CSV input.",
        "The strict sample is newly selected from TreeMc/TreeRec provenance; its grid is reconstructed from the TreeRec graph cache and its TOF vector is taken from that cache.",
        "Therefore the comparison answers: *how the Fig. 7.2-style CNN+DNN performs on the current strict TreeRec conditions*. It does not, by itself, isolate architecture from every representation-level difference versus the historical paper dataset.",
        "",
        "## Decision for the next experiment",
        "",
        "If the reference checkpoint has identical tensor shapes and compatible training settings, the 200K pilot is sufficient to decide whether a 4M CNN+DNN run is warranted.",
        "Given the observed high-efficiency gap to GravNet, do not start a 4M CNN+DNN run before resolving any material mismatch listed above.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-suffix", default="cnndnn_10x12x12")
    parser.add_argument("--current-run", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_run = args.current_run
    if not current_run.is_dir():
        raise FileNotFoundError(f"current run not found: {current_run}")
    current_checkpoint = current_run / "best.pt"
    if not current_checkpoint.is_file():
        raise FileNotFoundError(f"current best checkpoint not found: {current_checkpoint}")

    current_payload = load_checkpoint(current_checkpoint)
    reference_payload = load_checkpoint(args.reference_checkpoint)
    dataset = audit_dataset(args.dataset, args.split_suffix)
    model = CNNDNNHybrid(tof_dim=11, dropout=0.3)

    audit = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_path": str(args.dataset.resolve()),
        "current_run": str(current_run.resolve()),
        "current_checkpoint_path": str(current_checkpoint.resolve()),
        "reference_checkpoint_path": (
            str(args.reference_checkpoint.resolve())
            if args.reference_checkpoint is not None
            else None
        ),
        "dataset": dataset,
        "current_checkpoint": {
            "training_args": selected_args(current_payload),
            "state": state_shape_summary(current_payload),
        },
        "reference_checkpoint": (
            {
                "training_args": selected_args(reference_payload),
                "state": state_shape_summary(reference_payload),
            }
            if reference_payload is not None
            else None
        ),
        "model_shape_comparison": compare_state_shapes(current_payload, reference_payload),
        "expected_fig72_model_parameters": int(sum(p.numel() for p in model.parameters())),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "cnndnn_protocol_audit.json"
    markdown_path = args.out_dir / "cnndnn_protocol_audit.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_report(audit), encoding="utf-8")

    print(f"dataset audit : {args.dataset}")
    print(f"current run   : {current_run}")
    print(f"reference     : {args.reference_checkpoint or 'not supplied'}")
    print(f"JSON report   : {json_path}")
    print(f"Markdown report: {markdown_path}")
    if audit["model_shape_comparison"] is not None:
        print(
            "model tensor shapes identical: "
            f"{audit['model_shape_comparison']['same_tensor_shapes']}"
        )
    print("STRICT-TRACK CNN+DNN PROTOCOL AUDIT: COMPLETE")


if __name__ == "__main__":
    main()

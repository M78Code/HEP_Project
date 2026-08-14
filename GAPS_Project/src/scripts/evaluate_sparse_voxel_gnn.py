#!/usr/bin/env python3
"""Evaluate a sparse-voxel GNN/GravNet checkpoint.

The training script writes ``best.pt`` checkpoints that include the original
arguments and the TOF standardizer.  This evaluator restores those settings and
writes the same evaluation artifacts used by comparison scripts:

  labels.npy, scores.npy, betas.npy, metrics.json, rejection_curve.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

from GAPS_Project.src.scripts.train_aohba_sparse_voxel_gnn import (
    SparseVoxelDataset,
    SparseVoxelDGCNN,
    SparseVoxelGNN,
    SparseVoxelGravNet,
    evaluate,
)


def rejection_at(labels: np.ndarray, scores: np.ndarray, target: float) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    if len(candidates) == 0:
        return {
            "target_efficiency": target,
            "actual_efficiency": math.nan,
            "fpr": math.nan,
            "rejection": math.nan,
            "threshold": math.nan,
        }
    idx = candidates[np.argmin(fpr[candidates])]
    return {
        "target_efficiency": target,
        "actual_efficiency": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "rejection": math.inf if fpr[idx] == 0 else float(1.0 / fpr[idx]),
        "threshold": float(thresholds[idx]),
    }


def rejection_curve(labels: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=True)
    n_background = int((labels == 0).sum())
    fpr_floor = 1.0 / max(n_background, 1)
    return tpr, 1.0 / np.maximum(fpr, fpr_floor)


def build_model(train_args: SimpleNamespace, tof_dim: int, device: torch.device):
    model_name = getattr(train_args, "model", "graphconv")
    hidden = int(getattr(train_args, "hidden", 128))
    dropout = float(getattr(train_args, "dropout", 0.15))
    k = int(getattr(train_args, "k", 8))
    num_blocks = int(getattr(train_args, "num_blocks", 6))

    if model_name == "graphconv":
        model = SparseVoxelGNN(hidden=hidden, tof_dim=tof_dim, dropout=dropout)
    elif model_name == "gravnet":
        model = SparseVoxelGravNet(
            hidden=hidden,
            tof_dim=tof_dim,
            dropout=dropout,
            k=k,
            num_blocks=num_blocks,
        )
    elif model_name == "dgcnn":
        model = SparseVoxelDGCNN(
            hidden=hidden,
            tof_dim=tof_dim,
            dropout=dropout,
            k=k,
        )
    else:
        raise ValueError(f"unknown model in checkpoint args: {model_name}")
    return model.to(device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument(
        "--tof-mode",
        choices=["checkpoint", "paddles-primary", "primary"],
        default="checkpoint",
        help="Override checkpoint TOF mode. Default restores the mode saved during training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)

    train_args = SimpleNamespace(**ckpt.get("args", {}))
    data_dir = args.data_dir if args.data_dir is not None else Path(train_args.data_dir)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.model_path.parent / f"evaluation_{args.split}"
    )

    standardizer = ckpt.get("tof_standardizer", None)
    tof_mean = None if standardizer is None else standardizer.get("mean")
    tof_std = None if standardizer is None else standardizer.get("std")
    use_beta = bool(getattr(train_args, "use_beta", False))
    checkpoint_tof_mode = getattr(train_args, "tof_mode", "paddles-primary")
    tof_mode = checkpoint_tof_mode if args.tof_mode == "checkpoint" else args.tof_mode
    k = int(getattr(train_args, "k", 8))

    dataset = SparseVoxelDataset(
        data_dir,
        args.split,
        max_events=args.max_events,
        k=k,
        tof_mean=tof_mean,
        tof_std=tof_std,
        use_beta=use_beta,
        tof_mode=tof_mode,
    )
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    if args.num_workers > 0:
        loader_kw.update(dict(persistent_workers=True, prefetch_factor=2))
    loader = DataLoader(dataset, shuffle=False, **loader_kw)

    tof_dim = int(dataset.raw_tof(0).shape[0])
    model = build_model(train_args, tof_dim, device)
    state = ckpt.get("model", ckpt.get("model_state"))
    if state is None:
        raise KeyError("checkpoint has no model or model_state")
    model.load_state_dict(state)

    print("device:", device)
    print("data dir:", data_dir)
    print("split:", args.split)
    print("events:", len(dataset))
    print("batches:", len(loader))
    print("model:", getattr(train_args, "model", "graphconv"))
    print("use beta:", use_beta)
    print("tof mode:", tof_mode)
    print("tof/global dim:", tof_dim)
    print("checkpoint:", args.model_path)

    test_loss, test_acc, test_auc, labels, scores, betas = evaluate(
        model, loader, device, desc=args.split
    )

    metrics = {
        "n_events": int(labels.size),
        "label_counts": {
            "0": int((labels == 0).sum()),
            "1": int((labels == 1).sum()),
        },
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(labels.astype(int), (scores >= 0.5).astype(int))),
        "auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else math.nan,
        "rejection": [
            rejection_at(labels, scores, target)
            for target in (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
        ],
        "model_path": str(args.model_path),
        "data_dir": str(data_dir),
        "split": args.split,
        "use_beta": use_beta,
        "tof_mode": tof_mode,
        "tof_dim": tof_dim,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "scores.npy", scores)
    np.save(output_dir / "betas.npy", betas)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    if len(np.unique(labels)) == 2:
        tpr, rejection = rejection_curve(labels, scores)
        plt.figure(figsize=(7.0, 5.0), dpi=220)
        plt.plot(tpr, rejection, linewidth=2.0, label=f"AUC={metrics['auc']:.4f}")
        plt.yscale("log")
        plt.xlim(0.5, 1.0)
        plt.ylim(1, 1e6)
        plt.xlabel("Signal efficiency")
        plt.ylabel("Background rejection")
        plt.grid(True, which="both", linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "rejection_curve.png")
        plt.close()

    print("\nEVALUATION")
    print("loss:", test_loss)
    print("accuracy:", test_acc)
    print("AUC:", test_auc)
    for row in metrics["rejection"]:
        print(f"Rej@{row['target_efficiency']:.2f}: {row['rejection']}  FPR={row['fpr']}")
    print("saved:", output_dir)


if __name__ == "__main__":
    main()

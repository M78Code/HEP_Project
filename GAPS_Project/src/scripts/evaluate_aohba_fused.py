#!/usr/bin/env python
"""Evaluate FusedGravNet on Aohba mixed graph-cache shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from GAPS_Project.src.scripts.train_aohba_fused_amp import (
    AohbaFusedGravNet,
    ShardedGraphDataset,
    build_graph_feat,
    build_voxel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-tof172", action="store_true")
    parser.add_argument("--amp", action="store_true", help="use CUDA autocast during evaluation")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    return parser.parse_args()


def rejection_at_efficiency(labels: np.ndarray, scores: np.ndarray, target: float) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    idx = candidates[np.argmin(fpr[candidates])]
    rejection = float("inf") if fpr[idx] == 0 else 1.0 / fpr[idx]
    return {
        "target_efficiency": target,
        "actual_efficiency": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "rejection": float(rejection),
        "threshold": float(thresholds[idx]),
    }


def load_state(model_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
    else:
        state = checkpoint
    return {
        key.replace("_orig_mod.", "").replace("module.", ""): value
        for key, value in state.items()
    }


@torch.no_grad()
def infer(
    model: AohbaFusedGravNet,
    loader: DataLoader,
    device: torch.device,
    total_batches: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    model.eval()
    labels, scores, betas = [], [], []

    for batch in tqdm(loader, total=total_batches, desc="test", dynamic_ncols=True):
        batch = batch.to(device)
        graph_feat = build_graph_feat(batch)
        voxel = build_voxel(batch)
        autocast_enabled = amp_enabled and device.type == "cuda"
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
            logits = model(batch, graph_feat=graph_feat, voxel=voxel)
        probs = torch.softmax(logits, dim=1)[:, 1]

        labels.append(batch.y.view(-1).cpu().numpy())
        scores.append(probs.cpu().numpy())
        if hasattr(batch, "mc_beta"):
            betas.append(batch.mc_beta.view(-1).cpu().numpy())

    return (
        np.concatenate(labels),
        np.concatenate(scores),
        np.concatenate(betas) if betas else None,
    )


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ShardedGraphDataset(args.cache_dir, "test", seed=args.seed)
    n_events = dataset.approx_len()
    n_batches = (n_events + args.batch_size - 1) // args.batch_size
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = AohbaFusedGravNet(use_tof172=args.use_tof172).to(device)
    model.load_state_dict(load_state(args.model_path, device))

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_enabled = args.amp and device.type == "cuda"

    print(f"device     : {device}", flush=True)
    print(f"test files : {len(dataset.files)}", flush=True)
    print(f"test events: {n_events:,}", flush=True)
    print(f"model      : {args.model_path}", flush=True)
    print(f"use_tof172 : {args.use_tof172}", flush=True)
    print(f"AMP eval   : {amp_enabled} ({args.amp_dtype if amp_enabled else 'disabled'})", flush=True)

    labels, scores, betas = infer(
        model,
        loader,
        device,
        n_batches,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    predictions = (scores >= 0.5).astype(np.int64)
    metrics = {
        "n_events": int(len(labels)),
        "label_counts": {
            str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))
        },
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, scores)),
        "rejection": [
            rejection_at_efficiency(labels, scores, target)
            for target in (0.50, 0.80, 0.90, 0.95, 0.98, 0.99)
        ],
        "use_tof172": bool(args.use_tof172),
        "model_path": str(args.model_path),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "scores.npy", scores)
    if betas is not None:
        np.save(args.output_dir / "betas.npy", betas)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    fpr, tpr, _ = roc_curve(labels, scores)
    n_background = int((labels == 0).sum())
    plt.figure(figsize=(7, 6))
    plt.plot(
        tpr,
        1.0 / np.maximum(fpr, 1.0 / n_background),
        label=f"FusedGravNet (AUC={metrics['auc']:.4f})",
    )
    plt.yscale("log")
    plt.xlim(0.5, 1.0)
    plt.xlabel("Signal Efficiency (antiD recall)")
    plt.ylabel("Background Rejection (1 / FPR)")
    plt.grid(True, which="major", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "rejection_curve.png", dpi=300)
    plt.close()

    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"AUC     : {metrics['auc']:.6f}")
    for row in metrics["rejection"]:
        print(
            f"Rej@{row['target_efficiency']:.2f}: "
            f"{row['rejection']:.3f} "
            f"(actual eff={row['actual_efficiency']:.6f}, "
            f"FPR={row['fpr']:.8g})"
        )
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate CNN+DNN baseline on Aohba mixed graph-cache shards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from GAPS_Project.src.models.cnn_dnn_hybrid import CNNDNNHybrid
from GAPS_Project.src.scripts.train_aohba_cnndnn_amp import AohbaCNNDNNDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
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
def infer(model, loader, device, total_batches, amp_enabled: bool, amp_dtype: torch.dtype):
    model.eval()
    labels, scores, betas = [], [], []
    for voxel, tof, label in tqdm(loader, total=total_batches, desc="test", dynamic_ncols=True):
        voxel = voxel.to(device, non_blocking=True)
        tof = tof.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled and device.type == "cuda"):
            logits = model(voxel, tof)
        probs = torch.sigmoid(logits)
        labels.append(label.numpy())
        scores.append(probs.cpu().numpy())
        # The dataset yields tensors only. Load beta separately by iterating graph shards below.
    return np.concatenate(labels), np.concatenate(scores)


def load_betas(cache_dir: Path, seed: int) -> np.ndarray:
    values = []
    files = sorted(cache_dir.glob("test_mixed_*.pt"))
    random.Random(seed).shuffle(files)
    for pt_path in files:
        data_list = torch.load(pt_path, map_location="cpu", weights_only=False)
        for graph in data_list:
            if hasattr(graph, "mc_beta"):
                values.append(float(graph.mc_beta.view(()).item()))
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = AohbaCNNDNNDataset(args.cache_dir, "test", seed=args.seed)
    n_events = dataset.approx_len()
    n_batches = (n_events + args.batch_size - 1) // args.batch_size
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = CNNDNNHybrid(tof_dim=11).to(device)
    model.load_state_dict(load_state(args.model_path, device))

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_enabled = args.amp and device.type == "cuda"

    print(f"device     : {device}", flush=True)
    print(f"test files : {len(dataset.files)}", flush=True)
    print(f"test events: {n_events:,}", flush=True)
    print(f"model      : {args.model_path}", flush=True)
    print(f"AMP eval   : {amp_enabled} ({args.amp_dtype if amp_enabled else 'disabled'})", flush=True)

    labels, scores = infer(model, loader, device, n_batches, amp_enabled, amp_dtype)
    betas = load_betas(args.cache_dir, seed=args.seed)
    if len(betas) != len(labels):
        raise ValueError(f"betas length mismatch: {len(betas)} != {len(labels)}")

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
        "model_path": str(args.model_path),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "scores.npy", scores)
    np.save(args.output_dir / "betas.npy", betas)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    fpr, tpr, _ = roc_curve(labels, scores)
    n_background = int((labels == 0).sum())
    plt.figure(figsize=(7, 6))
    plt.plot(
        tpr,
        1.0 / np.maximum(fpr, 1.0 / n_background),
        label=f"CNN+DNN (AUC={metrics['auc']:.4f})",
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

#!/usr/bin/env python3
"""Train Fig.7.2-style CNN+DNN on exported Nakagami CSV arrays.

This script is intended for same-input comparison with the sparse/GravNet
GNN runs on Nakagami fixed-grid CSV data.  It uses only:

  - voxels.npy      : input to the CNN branch, Si(Li) voxel map
  - tof_primary.npy : input to the DNN branch, 11-D auxiliary/TOF feature vector

Supervised training uses:

  - labels.npy      : binary teacher label, 0=anti-proton, 1=anti-deuteron

It deliberately ignores beta so that the input matches the two-branch CNN+DNN
architecture shown in Nakagami thesis Fig. A.6 / Sec. 7.1.1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.cnn_dnn_hybrid import CNNDNNHybrid


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def split_dir(data_dir: Path, split: str) -> Path:
    return data_dir / f"{split}_nakagami_style_4M"


class NakagamiFig72CNNDNNDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        max_events: int | None = None,
    ) -> None:
        self.split_dir = split_dir(data_dir, split)
        if not self.split_dir.exists():
            raise FileNotFoundError(f"split directory not found: {self.split_dir}")

        self.voxels = np.load(self.split_dir / "voxels.npy", mmap_mode="r")
        self.tof = np.load(self.split_dir / "tof_primary.npy", mmap_mode="r")
        self.labels = np.load(self.split_dir / "labels.npy", mmap_mode="r")
        beta_path = self.split_dir / "betas.npy"
        self.betas = np.load(beta_path, mmap_mode="r") if beta_path.exists() else None

        if self.voxels.shape[1:] != (10, 12, 12):
            raise ValueError(f"voxel shape must be (N, 10, 12, 12), got {self.voxels.shape}")
        if self.tof.shape[1] != 11:
            raise ValueError(f"tof_primary dim must be 11, got {self.tof.shape[1]}")
        if len(self.voxels) != len(self.tof) or len(self.tof) != len(self.labels):
            raise ValueError("voxels/tof_primary/labels length mismatch")

        n_events = len(self.labels)
        if max_events is not None:
            n_events = min(n_events, int(max_events))
        self.n_events = n_events

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int):
        idx = int(idx)
        voxel = np.asarray(self.voxels[idx], dtype=np.float32).copy()
        tof = np.asarray(self.tof[idx], dtype=np.float32).copy()
        label = int(self.labels[idx])
        if self.betas is None:
            beta = np.nan
        else:
            beta = float(self.betas[idx])
        return (
            torch.from_numpy(voxel).unsqueeze(0),
            torch.from_numpy(tof),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(beta, dtype=torch.float32),
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    max_batches: int | None,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    desc = "train" if train else "val"
    for batch_idx, (voxel, tof, labels, _betas) in enumerate(
        tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

        voxel = voxel.to(device, non_blocking=True)
        tof = tof.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_float = labels.float()

        autocast_enabled = amp_enabled and device.type == "cuda"
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                logits = model(voxel, tof)
                loss = criterion(logits, labels_float)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                    logits = model(voxel, tof)
                    loss = criterion(logits, labels_float)

        total_loss += float(loss.item()) * labels.numel()
        preds = (torch.sigmoid(logits) >= 0.5).long()
        total_correct += int((preds == labels).sum().item())
        total_samples += int(labels.numel())

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def infer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    model.eval()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    betas_all: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.BCEWithLogitsLoss()

    for voxel, tof, labels, betas in tqdm(loader, desc="test", leave=False, dynamic_ncols=True):
        voxel = voxel.to(device, non_blocking=True)
        tof = tof.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_float = labels.float()

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled and device.type == "cuda"):
            logits = model(voxel, tof)
            loss = criterion(logits, labels_float)
        scores = torch.sigmoid(logits)

        labels_all.append(labels.cpu().numpy())
        scores_all.append(scores.cpu().numpy())
        betas_all.append(betas.numpy())
        total_loss += float(loss.item()) * labels.numel()
        total_samples += int(labels.numel())

    labels_np = np.concatenate(labels_all).astype(np.int64)
    scores_np = np.concatenate(scores_all).astype(np.float32)
    betas_np = np.concatenate(betas_all).astype(np.float32)
    return total_loss / max(total_samples, 1), labels_np, scores_np, betas_np


def save_evaluation(
    out_dir: Path,
    model_path: Path,
    labels: np.ndarray,
    scores: np.ndarray,
    betas: np.ndarray,
    test_loss: float,
) -> dict:
    predictions = (scores >= 0.5).astype(np.int64)
    metrics = {
        "n_events": int(labels.size),
        "label_counts": {
            "0": int((labels == 0).sum()),
            "1": int((labels == 1).sum()),
        },
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, scores)),
        "rejection": [
            rejection_at(labels, scores, target)
            for target in (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
        ],
        "model_path": str(model_path),
        "input": {
            "voxel": "voxels.npy",
            "tof": "tof_primary.npy",
            "ignored": ["betas.npy"],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "scores.npy", scores)
    np.save(out_dir / "betas.npy", betas)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    tpr, rejection = rejection_curve(labels, scores)
    plt.figure(figsize=(7.0, 5.0), dpi=220)
    plt.plot(tpr, rejection, linewidth=2.0, label=f"CNN+DNN AUC={metrics['auc']:.4f}")
    plt.yscale("log")
    plt.xlim(0.5, 1.0)
    plt.ylim(1, 1e6)
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rejection_curve.png")
    plt.close()

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset-tag", default="nakagami_fig72_cnndnn")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--step-size", type=int, default=999999)
    p.add_argument("--lr-gamma", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--max-train-events", type=int, default=None)
    p.add_argument("--max-val-events", type=int, default=None)
    p.add_argument("--max-test-events", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint saved as last.pt or best.pt.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ds = NakagamiFig72CNNDNNDataset(args.data_dir, "train", args.max_train_events)
    val_ds = NakagamiFig72CNNDNNDataset(args.data_dir, "val", args.max_val_events)
    test_ds = NakagamiFig72CNNDNNDataset(args.data_dir, "test", args.max_test_events)

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    model = CNNDNNHybrid(tof_dim=11, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_gamma)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    if args.resume is not None:
        run_dir = args.resume.resolve().parent
    else:
        run_dir = (
            PROJECT_ROOT
            / "results"
            / f"{datetime.now():%Y%m%d-%H%M%S}_CNNDNNFig72_{args.dataset_tag}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        no_improve = int(ckpt.get("no_improve", no_improve))

    print(f"device     : {device}", flush=True)
    print(f"data dir   : {args.data_dir}", flush=True)
    print(f"train/val/test: {len(train_ds):,} {len(val_ds):,} {len(test_ds):,}", flush=True)
    print(f"batches    : {len(train_loader):,} {len(val_loader):,} {len(test_loader):,}", flush=True)
    print("input      : voxels.npy + tof_primary.npy (beta ignored)", flush=True)
    print("model      : CNNDNNHybrid / Nakagami Fig.7.2 style", flush=True)
    print(f"parameters : {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"AMP        : {amp_enabled} ({args.amp_dtype if amp_enabled else 'disabled'})", flush=True)
    if args.resume is not None:
        print(f"resume     : {args.resume}", flush=True)
        print(f"start epoch: {start_epoch}", flush=True)
    print(
        "early stopping: "
        f"monitor=val_loss mode=min min_epochs={args.min_epochs} "
        f"patience={args.patience} min_delta={args.min_delta}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True,
            args.max_train_batches,
            amp_enabled,
            amp_dtype,
            scaler,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            False,
            args.max_val_batches,
            amp_enabled,
            amp_dtype,
            None,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "args": vars(args),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
            },
            last_path,
        )

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "no_improve": no_improve,
                },
                best_path,
            )
            print(f"  -> best saved: {best_path} (val_loss={best_val_loss:.6f})", flush=True)
        else:
            no_improve += 1

        if args.patience > 0 and epoch >= args.min_epochs and no_improve >= args.patience:
            print(
                "early stopping triggered: "
                f"epoch={epoch}, best_epoch={best_epoch}, "
                f"best_val_loss={best_val_loss:.6f}, no_improve_epochs={no_improve}",
                flush=True,
            )
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_loss, labels, scores, betas = infer(model, test_loader, device, amp_enabled, amp_dtype)
    eval_dir = run_dir / "evaluation_test"
    metrics = save_evaluation(eval_dir, best_path, labels, scores, betas, test_loss)

    print("\nTEST", flush=True)
    print(f"loss    : {metrics['test_loss']}", flush=True)
    print(f"accuracy: {metrics['accuracy']}", flush=True)
    print(f"AUC     : {metrics['auc']}", flush=True)
    for row in metrics["rejection"]:
        print(
            f"Rej@{row['target_efficiency']:.2f}: {row['rejection']}  FPR={row['fpr']}",
            flush=True,
        )
    print(f"saved: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

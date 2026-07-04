#!/usr/bin/env python
"""Train Nakagami-style three-input CNN+DNN on exported Aohba 4M arrays."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import GAPS_Project
from GAPS_Project.src.models.nakagami_three_input import NakagamiThreeInputNet


PROJECT_ROOT = Path(GAPS_Project.__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="aohba4M_nakagami3input")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--step-size", type=int, default=15)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-events", type=int, default=None)
    parser.add_argument("--max-val-events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


class NakagamiStyle3InputDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, max_events: int | None = None):
        self.split_dir = data_dir / f"{split}_nakagami_style_4M"
        if not self.split_dir.exists():
            raise FileNotFoundError(f"split directory not found: {self.split_dir}")

        self.voxels = np.load(self.split_dir / "voxels.npy", mmap_mode="r")
        self.tof_paddles = np.load(self.split_dir / "tof_paddles.npy", mmap_mode="r")
        self.tof_features = np.load(self.split_dir / "tof_primary.npy", mmap_mode="r")
        self.labels = np.load(self.split_dir / "labels.npy", mmap_mode="r")

        if self.voxels.shape[1:] != (10, 12, 12):
            raise ValueError(f"voxel shape must be (N, 10, 12, 12), got {self.voxels.shape}")
        if self.tof_paddles.shape[1] != 172:
            raise ValueError(f"tof_paddles dim must be 172, got {self.tof_paddles.shape[1]}")
        if self.tof_features.shape[1] != 11:
            raise ValueError(f"tof_primary/tof_features dim must be 11, got {self.tof_features.shape[1]}")

        n_events = len(self.labels)
        if max_events is not None:
            n_events = min(n_events, int(max_events))
        self.n_events = n_events

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int):
        idx = int(idx)
        voxel = np.asarray(self.voxels[idx], dtype=np.float32).copy()
        tof_paddle = np.asarray(self.tof_paddles[idx], dtype=np.float32).copy()
        tof_feature = np.asarray(self.tof_features[idx], dtype=np.float32).copy()
        label = int(self.labels[idx])
        return (
            torch.from_numpy(voxel).unsqueeze(0),
            torch.from_numpy(tof_paddle),
            torch.from_numpy(tof_feature),
            torch.tensor(label, dtype=torch.long),
        )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
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
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for batch_idx, (voxel, tof_paddle, tof_feature, labels) in enumerate(bar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        voxel = voxel.to(device, non_blocking=True)
        tof_paddle = tof_paddle.to(device, non_blocking=True)
        tof_feature = tof_feature.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_float = labels.float()

        autocast_enabled = amp_enabled and device.type == "cuda"
        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                logits = model(voxel, tof_paddle, tof_feature)
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
                    logits = model(voxel, tof_paddle, tof_feature)
                    loss = criterion(logits, labels_float)

        total_loss += float(loss.item()) * labels.numel()
        preds = (torch.sigmoid(logits) >= 0.5).long()
        total_correct += int((preds == labels).sum().item())
        total_samples += int(labels.numel())
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples, total_correct / total_samples, total_samples


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = NakagamiStyle3InputDataset(args.data_dir, "train", args.max_train_events)
    val_dataset = NakagamiStyle3InputDataset(args.data_dir, "val", args.max_val_events)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    train_batches = args.max_train_batches or ((len(train_dataset) + args.batch_size - 1) // args.batch_size)
    val_batches = args.max_val_batches or ((len(val_dataset) + args.batch_size - 1) // args.batch_size)

    model = NakagamiThreeInputNet(dropout_res=0.1, dropout_dense=0.2).to(device)
    model_name = f"NakagamiThreeInputAMP_10x12x12_{args.dataset_tag}"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    print(f"device     : {device}", flush=True)
    print(f"data dir   : {args.data_dir}", flush=True)
    print(f"train events: {len(train_dataset):,}  batches: {train_batches:,}", flush=True)
    print(f"val   events: {len(val_dataset):,}  batches: {val_batches:,}", flush=True)
    print(f"model      : {model_name}", flush=True)
    print(f"parameters : {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"AMP        : {amp_enabled} ({args.amp_dtype if amp_enabled else 'disabled'})", flush=True)
    print(f"early stopping: min_epochs={args.min_epochs}, patience={args.patience}, monitor=val_loss", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_gamma)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.resume is not None:
        log_dir = args.resume.parent
    else:
        log_dir = PROJECT_ROOT / "results" / f"{timestamp}_{model_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float("inf")
    best_model_path = log_dir / f"{timestamp}_{model_name}_best.pth"
    latest_path = log_dir / f"{timestamp}_{model_name}_last_checkpoint.pth"
    patience_counter = 0
    start_epoch = 1

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        patience_counter = int(checkpoint.get("patience_counter", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        latest_path = args.resume
        best_matches = sorted(log_dir.glob("*_best.pth"))
        if best_matches:
            best_model_path = best_matches[0]
        print(
            f"resumed checkpoint: {args.resume} "
            f"(next epoch={start_epoch}, best_val_loss={best_val_loss:.4f}, "
            f"patience_counter={patience_counter})",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_loss, train_acc, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, True,
            args.max_train_batches, amp_enabled, amp_dtype, scaler,
        )
        val_loss, val_acc, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, False,
            args.max_val_batches, amp_enabled, amp_dtype, None,
        )
        scheduler.step()
        elapsed = time.time() - started

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s",
            flush=True,
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        checkpoint = {
            "epoch": epoch,
            "model_name": "nakagami_three_input",
            "dataset_tag": args.dataset_tag,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "args": vars(args),
            "amp_enabled": amp_enabled,
            "amp_dtype": args.amp_dtype if amp_enabled else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> best model saved (val_loss={best_val_loss:.4f})", flush=True)
        else:
            patience_counter += 1

        checkpoint["best_val_loss"] = best_val_loss
        checkpoint["patience_counter"] = patience_counter
        torch.save(checkpoint, latest_path)
        print(f"  -> latest checkpoint saved: {latest_path}", flush=True)

        if epoch >= args.min_epochs and patience_counter >= args.patience:
            print(f"  -> early stopping: val_loss no improvement for {args.patience} epochs", flush=True)
            break

    writer.close()
    print(f"\ntraining complete, best model: {best_model_path}", flush=True)


if __name__ == "__main__":
    train(parse_args())

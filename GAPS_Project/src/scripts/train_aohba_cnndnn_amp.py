#!/usr/bin/env python
"""Train CNN+DNN baseline on Aohba mixed graph-cache shards with AMP.

This script uses the same 4M fused mixed cache as FusedGravNet.  It reads
per-event Si(Li) voxels and TreeRec TOF features from each graph object, so the
result is comparable with GravNet, DGCNN, and FusedGravNet on the same split.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import GAPS_Project
from GAPS_Project.src.models.cnn_dnn_hybrid import CNNDNNHybrid


PROJECT_ROOT = Path(GAPS_Project.__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="aohba4M_cnndnn")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def count_graphs(pt_path: Path) -> int:
    json_path = pt_path.with_suffix(".json")
    if json_path.exists():
        with json_path.open(encoding="utf-8") as handle:
            return int(json.load(handle)["n_graphs"])
    return len(torch.load(pt_path, map_location="cpu", weights_only=False))


class AohbaCNNDNNDataset(IterableDataset):
    def __init__(self, cache_dir: Path, split: str, seed: int = 42):
        super().__init__()
        self.cache_dir = cache_dir
        self.split = split
        self.seed = seed
        self.files = sorted(cache_dir.glob(f"{split}_mixed_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no {split}_mixed_*.pt files in {cache_dir}")
        self._approx_len = sum(count_graphs(path) for path in self.files)

    def approx_len(self) -> int:
        return self._approx_len

    def __iter__(self):
        files = list(self.files)
        random.Random(self.seed).shuffle(files)
        for pt_path in files:
            data_list = torch.load(pt_path, map_location="cpu", weights_only=False)
            for graph in data_list:
                if not hasattr(graph, "voxel"):
                    raise RuntimeError(f"voxel missing in graph-cache shard: {pt_path}")
                if not hasattr(graph, "tof_feat"):
                    raise RuntimeError(f"tof_feat missing in graph-cache shard: {pt_path}")
                voxel = torch.log1p(graph.voxel.float()).view(1, 10, 20, 20)
                tof = graph.tof_feat.float().view(11)
                label = graph.y.view(()).long()
                yield voxel, tof, label

    def __len__(self) -> int:
        return self._approx_len


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
    bar = tqdm(loader, desc=desc, leave=False)

    for batch_idx, (voxel, tof, labels) in enumerate(bar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        voxel = voxel.to(device, non_blocking=True)
        tof = tof.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_float = labels.float()

        autocast_enabled = amp_enabled and device.type == "cuda"
        if train:
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
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples, total_correct / total_samples, total_samples


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = AohbaCNNDNNDataset(args.split_cache_dir, "train", seed=args.seed)
    val_dataset = AohbaCNNDNNDataset(args.split_cache_dir, "val", seed=args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_batches = args.max_train_batches or ((train_dataset.approx_len() + args.batch_size - 1) // args.batch_size)
    val_batches = args.max_val_batches or ((val_dataset.approx_len() + args.batch_size - 1) // args.batch_size)

    model = CNNDNNHybrid(tof_dim=11).to(device)
    model_name = f"CNNDNNHybridAMP_10x20x20_{args.dataset_tag}"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    print(f"device     : {device}", flush=True)
    print(f"split cache: {args.split_cache_dir}", flush=True)
    print(f"train events (approx): {train_dataset.approx_len():,}  batches: {train_batches:,}", flush=True)
    print(f"val   events (approx): {val_dataset.approx_len():,}  batches: {val_batches:,}", flush=True)
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
            "model_name": "cnn_dnn_hybrid",
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

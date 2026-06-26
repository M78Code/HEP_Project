#!/usr/bin/env python
"""Train FusedGravNet with AMP on Aohba mixed graph-cache shards."""

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
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GravNetConv, global_mean_pool
from tqdm import tqdm

import GAPS_Project
from GAPS_Project.src.losses import FocalLoss
from GAPS_Project.src.models.fused_model import CNNEncoder


PROJECT_ROOT = Path(GAPS_Project.__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="aohba4M_fused")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--step-size", type=int, default=15)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-tof172", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="disable CUDA automatic mixed precision")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def count_graphs(pt_path: Path) -> int:
    json_path = pt_path.with_suffix(".json")
    if json_path.exists():
        with json_path.open() as f:
            row = json.load(f)
        return int(row["n_graphs"])
    return len(torch.load(pt_path, map_location="cpu", weights_only=False))


class ShardedGraphDataset(IterableDataset):
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
                yield graph

    def __len__(self) -> int:
        return self._approx_len


class AohbaFusedGravNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 128,
        space_dimensions: int = 4,
        propagate_dimensions: int = 22,
        k: int = 8,
        num_classes: int = 2,
        dropout: float = 0.3,
        graph_feat_dim: int = 45,
        num_blocks: int = 6,
        cnn_feat_dim: int = 128,
        use_tof172: bool = False,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.use_tof172 = use_tof172

        self.pre_linears = nn.ModuleList()
        self.gravnet_layers = nn.ModuleList()
        self.post_norms = nn.ModuleList()

        current_dim = in_channels
        for _ in range(num_blocks):
            self.pre_linears.append(nn.Sequential(nn.Linear(current_dim, hidden_dim), nn.Tanh()))
            self.gravnet_layers.append(
                GravNetConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    space_dimensions=space_dimensions,
                    propagate_dimensions=propagate_dimensions,
                    k=k,
                )
            )
            self.post_norms.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim

        self.skip_linear = nn.Linear(in_channels, hidden_dim)
        self.cnn_encoder = CNNEncoder(dropout=dropout)

        tof_feat_dim = 0
        if use_tof172:
            tof_feat_dim = 64
            self.tof_branch = nn.Sequential(
                nn.Linear(172, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, tof_feat_dim),
                nn.ReLU(),
            )

        gravnet_out_dim = hidden_dim * num_blocks + hidden_dim
        concat_dim = gravnet_out_dim + graph_feat_dim + cnn_feat_dim + tof_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch, graph_feat: torch.Tensor, voxel: torch.Tensor) -> torch.Tensor:
        x = batch.x
        batch_index = batch.batch

        x_skip = self.skip_linear(x)
        block_outputs = []
        x_cur = x
        for pre_linear, gravnet, norm in zip(self.pre_linears, self.gravnet_layers, self.post_norms):
            x_cur = pre_linear(x_cur)
            x_cur = gravnet(x_cur, batch=batch_index)
            x_cur = norm(x_cur).relu()
            block_outputs.append(x_cur)

        x_cat = torch.cat(block_outputs + [x_skip], dim=1)
        graph_embedding = global_mean_pool(x_cat, batch_index)
        cnn_embedding = self.cnn_encoder(voxel)

        parts = [graph_embedding, graph_feat, cnn_embedding]
        if self.use_tof172:
            if not hasattr(batch, "tof_paddle_energy"):
                raise RuntimeError("batch.tof_paddle_energy missing, but --use-tof172 was set")
            tof = batch.tof_paddle_energy.view(-1, 172)
            parts.append(self.tof_branch(tof))

        return self.classifier(torch.cat(parts, dim=1))


def build_graph_feat(batch) -> torch.Tensor:
    return torch.cat(
        [
            batch.n_hits.view(-1, 1),
            batch.total_energy.view(-1, 1),
            batch.sili_profile.view(-1, 16),
            batch.tof_profile.view(-1, 16),
            batch.tof_feat.view(-1, 11),
        ],
        dim=1,
    )


def build_voxel(batch) -> torch.Tensor:
    return torch.log1p(batch.voxel.view(-1, 1, 10, 20, 20))


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
    for batch_idx, batch in enumerate(bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = batch.to(device)
        labels = batch.y.view(-1)
        graph_feat = build_graph_feat(batch)
        voxel = build_voxel(batch)

        autocast_enabled = amp_enabled and device.type == "cuda"
        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=autocast_enabled
            ):
                logits = model(batch, graph_feat=graph_feat, voxel=voxel)
                loss = criterion(logits, labels)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda", dtype=amp_dtype, enabled=autocast_enabled
                ):
                    logits = model(batch, graph_feat=graph_feat, voxel=voxel)
                    loss = criterion(logits, labels)

        total_loss += float(loss.item()) * batch.num_graphs
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_samples += int(batch.num_graphs)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples, total_correct / total_samples, total_samples


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_dataset = ShardedGraphDataset(args.split_cache_dir, "train", seed=args.seed)
    val_dataset = ShardedGraphDataset(args.split_cache_dir, "val", seed=args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_batches = args.max_train_batches or ((train_dataset.approx_len() + args.batch_size - 1) // args.batch_size)
    val_batches = args.max_val_batches or ((val_dataset.approx_len() + args.batch_size - 1) // args.batch_size)

    model = AohbaFusedGravNet(use_tof172=args.use_tof172).to(device)
    model_tag = "FusedGravNetTOF" if args.use_tof172 else "FusedGravNet"
    model_name = f"{model_tag}AMP_6b_h128_{args.dataset_tag}"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )

    print(f"device     : {device}", flush=True)
    print(f"split cache: {args.split_cache_dir}", flush=True)
    print(f"train events (approx): {train_dataset.approx_len():,}  batches: {train_batches:,}", flush=True)
    print(f"val   events (approx): {val_dataset.approx_len():,}  batches: {val_batches:,}", flush=True)
    print(f"model      : {model_name}", flush=True)
    print(f"parameters : {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(
        f"AMP        : {amp_enabled} "
        f"(dtype={args.amp_dtype if amp_enabled else 'disabled'})",
        flush=True,
    )
    print(f"early stopping: min_epochs={args.min_epochs}, patience={args.patience}, monitor=val_loss", flush=True)

    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_gamma)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / "results" / f"{timestamp}_{model_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float("inf")
    best_model_path = log_dir / f"{timestamp}_{model_name}_best.pth"
    latest_path = log_dir / f"{timestamp}_{model_name}_last_checkpoint.pth"
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_acc, _ = run_epoch(
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
        val_loss, val_acc, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            False,
            args.max_val_batches,
            amp_enabled,
            amp_dtype,
            None,
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
            "model_name": "fused_gravnet_tof" if args.use_tof172 else "fused_gravnet",
            "dataset_tag": args.dataset_tag,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "args": vars(args),
            "amp_enabled": amp_enabled,
            "amp_dtype": args.amp_dtype if amp_enabled else None,
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

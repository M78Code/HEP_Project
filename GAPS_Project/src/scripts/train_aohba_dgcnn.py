#!/usr/bin/env python
"""Train DGCNN on Aohba split graph-cache shards.

This script is intentionally separate from train_aohba.py so DGCNN can be
run as a supplementary 4M balanced-subset baseline without touching the
main GravNet/GravNetTOF training path.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import GAPS_Project
from GAPS_Project.src.losses import FocalLoss
from GAPS_Project.src.models.dgcnn import DGCNNClassifier


PROJECT_ROOT = Path(GAPS_Project.__file__).parent

EPOCHS = 80
BATCH_SIZE = 128
LR = 3e-4
STEP_SIZE = 15
GAMMA = 0.5
FOCAL_GAMMA = 1.5
PATIENCE = 10
MIN_EPOCHS = 20
IN_CHANNEL = 8
HIDDEN_DIM = 64


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CachedStreamDataset(IterableDataset):
    def __init__(
        self,
        pt_files: list[Path],
        shuffle_files: bool = True,
        shuffle_events: bool = True,
        seed: int = 42,
        balance_tagged_classes: bool = False,
    ):
        self.pt_files = list(pt_files)
        self.shuffle_files = shuffle_files
        self.shuffle_events = shuffle_events
        self.seed = seed
        self.balance_tagged_classes = balance_tagged_classes
        self._epoch = 0

    @staticmethod
    def _load_shard(pt_path: Path):
        try:
            data_list = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError(
                f"failed to load graph-cache shard: {pt_path} "
                f"({type(error).__name__}: {error})"
            ) from error
        if not isinstance(data_list, list) or not data_list:
            raise RuntimeError(
                f"invalid graph-cache shard: {pt_path} "
                f"(expected a non-empty list, got {type(data_list).__name__})"
            )
        return data_list

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1

        if self.balance_tagged_classes:
            antiD_files = [path for path in self.pt_files if "_antiD_" in path.name]
            antiP_files = [path for path in self.pt_files if "_antiP_" in path.name]
            if antiD_files and antiP_files:
                yield from self._iter_balanced(antiD_files, antiP_files, epoch)
                return

        files = self.pt_files.copy()
        if self.shuffle_files:
            random.Random(self.seed + epoch).shuffle(files)

        for file_index, pt_path in enumerate(files):
            data_list = self._load_shard(pt_path)
            if self.shuffle_events:
                random.Random(
                    self.seed + epoch * 1_000_003 + file_index
                ).shuffle(data_list)
            yield from data_list

    def _iter_balanced(self, antiD_files, antiP_files, epoch):
        antiD_files = antiD_files.copy()
        antiP_files = antiP_files.copy()
        if self.shuffle_files:
            random.Random(self.seed + epoch).shuffle(antiD_files)
            random.Random(self.seed + epoch + 10_000).shuffle(antiP_files)

        def class_stream(files, class_offset):
            for file_index, pt_path in enumerate(files):
                data_list = self._load_shard(pt_path)
                if self.shuffle_events:
                    random.Random(
                        self.seed
                        + epoch * 1_000_003
                        + class_offset
                        + file_index
                    ).shuffle(data_list)
                yield from data_list

        antiD_stream = class_stream(antiD_files, 100_000)
        antiP_stream = class_stream(antiP_files, 200_000)
        for antiD_graph, antiP_graph in zip(antiD_stream, antiP_stream):
            yield antiD_graph
            yield antiP_graph

    def approx_len(self) -> int:
        def count_files(files):
            total = 0
            for pt_path in files:
                summary = Path(pt_path).with_suffix(".json")
                if summary.exists():
                    with summary.open(encoding="utf-8") as f:
                        total += int(json.load(f).get("n_graphs", 0))
                else:
                    total += len(self._load_shard(pt_path))
            return total

        if self.balance_tagged_classes:
            antiD_files = [path for path in self.pt_files if "_antiD_" in path.name]
            antiP_files = [path for path in self.pt_files if "_antiP_" in path.name]
            if antiD_files and antiP_files:
                return 2 * min(count_files(antiD_files), count_files(antiP_files))
        return count_files(self.pt_files)


def find_split_files(split_cache_dir: Path, split: str) -> list[Path]:
    files = sorted(split_cache_dir.glob(f"{split}_*.pt"))
    if files:
        return files
    single = split_cache_dir / f"{split}.pt"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no {split}_*.pt or {split}.pt found under {split_cache_dir}")


def make_loaders(split_cache_dir: Path, batch_size: int, seed: int):
    train_ds = CachedStreamDataset(
        find_split_files(split_cache_dir, "train"),
        shuffle_files=True,
        shuffle_events=True,
        seed=seed,
        balance_tagged_classes=True,
    )
    val_ds = CachedStreamDataset(
        find_split_files(split_cache_dir, "val"),
        shuffle_files=False,
        shuffle_events=False,
        seed=seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader, train_ds, val_ds


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


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device()
    print(f"使用设备：{device}")
    print(f"split cache: {args.split_cache_dir}")

    train_loader, val_loader, train_ds, val_ds = make_loaders(
        args.split_cache_dir, args.batch_size, args.seed
    )
    train_approx = train_ds.approx_len()
    val_approx = val_ds.approx_len()
    train_batches = (train_approx + args.batch_size - 1) // args.batch_size
    val_batches = (val_approx + args.batch_size - 1) // args.batch_size
    if args.max_train_batches is not None:
        train_batches = min(train_batches, args.max_train_batches)
    if args.max_val_batches is not None:
        val_batches = min(val_batches, args.max_val_batches)
    print(f"train events (approx): {train_approx:,}  batches: {train_batches:,}")
    print(f"val   events (approx): {val_approx:,}  batches: {val_batches:,}")

    dataset_tag = args.dataset_tag or args.split_cache_dir.name
    exp_name = f"DGCNN_h{args.hidden_dim}_{dataset_tag}"
    model = DGCNNClassifier(
        in_channels=IN_CHANNEL,
        hidden_dim=args.hidden_dim,
        k=args.k,
        graph_feat_dim=45,
    ).to(device)
    print(f"模型: {exp_name}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"early stopping: min_epochs={args.min_epochs}, "
        f"patience={args.patience}, monitor=val_loss"
    )

    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_gamma)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_root = args.result_dir or PROJECT_ROOT / "results"
    log_dir = result_root / f"{timestamp}_{exp_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    best_model_path = log_dir / f"{timestamp}_{exp_name}_best.pth"
    latest_checkpoint_path = log_dir / f"{timestamp}_{exp_name}_last_checkpoint.pth"

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{args.epochs} [train]",
            total=train_batches,
            leave=False,
        )
        for batch_idx, batch in enumerate(train_bar):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            batch = batch.to(device)
            labels = batch.y.view(-1)
            graph_feat = build_graph_feat(batch)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch.num_graphs
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += int(batch.num_graphs)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch:3d}/{args.epochs} [val]  ",
                total=val_batches,
                leave=False,
            )
            for batch_idx, batch in enumerate(val_bar):
                if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                    break
                batch = batch.to(device)
                labels = batch.y.view(-1)
                graph_feat = build_graph_feat(batch)
                logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
                loss = criterion(logits, labels)
                val_loss += float(loss.item()) * batch.num_graphs
                val_correct += int((logits.argmax(dim=1) == labels).sum().item())
                val_samples += int(batch.num_graphs)

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        scheduler.step()
        elapsed = time.time() - epoch_start

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

        should_stop = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience and epoch >= args.min_epochs:
                should_stop = True

        torch.save(
            {
                "epoch": epoch,
                "model_name": "dgcnn",
                "dataset_tag": dataset_tag,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "best_model_path": str(best_model_path),
                "args": vars(args),
            },
            latest_checkpoint_path,
        )
        print(f"  -> latest checkpoint saved: {latest_checkpoint_path}")

        if should_stop:
            print(f"  -> early stopping: val_loss no improvement for {args.patience} epochs")
            break

    writer.close()
    print(f"\n训练完成，最优模型: {best_model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--step-size", type=int, default=STEP_SIZE)
    parser.add_argument("--lr-gamma", type=float, default=GAMMA)
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

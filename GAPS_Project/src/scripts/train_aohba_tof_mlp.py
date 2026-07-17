#!/usr/bin/env python3
"""TOF-only neural baseline for Nakagami-style exported arrays.

The exported directory must contain:

  <data-dir>/<split>_nakagami_style_4M/
    tof_paddles.npy
    tof_primary.npy
    labels.npy
    betas.npy

This is a quick diagnostic baseline.  It checks whether a small neural network
can recover the strong information seen in the sklearn TOF/global-feature
baseline before spending more time on GNN fusion models.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


def split_dir(data_dir: str | Path, split: str) -> Path:
    return Path(data_dir) / f"{split}_nakagami_style_4M"


class TofDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str, mode: str, max_events: int | None = None):
        d = split_dir(data_dir, split)
        if not d.exists():
            raise FileNotFoundError(d)

        self.tof_paddles = np.load(d / "tof_paddles.npy", mmap_mode="r")
        self.tof_primary = np.load(d / "tof_primary.npy", mmap_mode="r")
        self.labels = np.load(d / "labels.npy", mmap_mode="r")
        self.betas = np.load(d / "betas.npy", mmap_mode="r")
        self.mode = mode

        n = len(self.labels)
        if max_events is not None:
            n = min(n, int(max_events))
        self.n = n

        if mode == "primary":
            self.dim = self.tof_primary.shape[1]
        elif mode == "paddles":
            self.dim = self.tof_paddles.shape[1]
        elif mode == "all":
            self.dim = self.tof_paddles.shape[1] + self.tof_primary.shape[1]
        elif mode == "primary_beta":
            self.dim = self.tof_primary.shape[1] + 1
        else:
            raise ValueError(f"unknown mode: {mode}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        idx = int(idx)
        primary = np.asarray(self.tof_primary[idx], dtype=np.float32)
        paddles = np.asarray(self.tof_paddles[idx], dtype=np.float32)
        beta = np.asarray([self.betas[idx]], dtype=np.float32)

        primary = np.log1p(np.clip(primary, 0, None)).astype(np.float32)
        paddles = np.log1p(np.clip(paddles, 0, None)).astype(np.float32)

        if self.mode == "primary":
            x = primary
        elif self.mode == "paddles":
            x = paddles
        elif self.mode == "all":
            x = np.concatenate([paddles, primary], axis=0)
        elif self.mode == "primary_beta":
            x = np.concatenate([primary, beta], axis=0)
        else:
            raise ValueError(self.mode)

        y = np.asarray([self.labels[idx]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(float(self.betas[idx]))


class TofMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def rejection_at_eff(labels: np.ndarray, scores: np.ndarray, eff: float):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    sig = scores[labels == 1]
    bkg = scores[labels == 0]
    if len(sig) == 0 or len(bkg) == 0:
        return float("nan"), float("nan")
    thr = np.quantile(sig, 1.0 - eff)
    fpr = float(np.mean(bkg >= thr))
    return (float("inf"), 0.0) if fpr == 0 else (1.0 / fpr, fpr)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ss, bs = [], [], []
    total_loss, total_n = 0.0, 0
    for x, y, beta in loader:
        x = x.to(device)
        y = y.view(-1).to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        prob = torch.sigmoid(logits)

        ys.append(y.cpu().numpy())
        ss.append(prob.cpu().numpy())
        bs.append(beta.numpy())
        total_loss += float(loss.item()) * y.numel()
        total_n += y.numel()

    y = np.concatenate(ys)
    s = np.concatenate(ss)
    b = np.concatenate(bs)
    acc = accuracy_score(y.astype(int), (s >= 0.5).astype(int))
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
    return total_loss / max(total_n, 1), acc, auc, y, s, b


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path("results") / f"{datetime.now():%Y%m%d-%H%M%S}_TofMLP_{args.dataset_tag}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)

    train_ds = TofDataset(args.data_dir, "train", args.mode, args.max_train_events)
    val_ds = TofDataset(args.data_dir, "val", args.mode, args.max_val_events)
    test_ds = TofDataset(args.data_dir, "test", args.mode, args.max_test_events)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if args.num_workers > 0:
        loader_kw.update(dict(persistent_workers=True, prefetch_factor=2))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    model = TofMLP(train_ds.dim, args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print("device:", device)
    print("data:", args.data_dir)
    print("mode:", args.mode, "input_dim:", train_ds.dim)
    print("train/val/test:", len(train_ds), len(val_ds), len(test_ds))
    print("batches:", len(train_loader), len(val_loader), len(test_loader))
    print("model params:", sum(p.numel() for p in model.parameters()))

    best_val = float("inf")
    best_path = out / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for bi, (x, y, _) in enumerate(train_loader):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            x = x.to(device)
            y = y.view(-1).to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += int((pred == y).sum().item())
            total_loss += float(loss.item()) * y.numel()
            total_n += y.numel()

        val_loss, val_acc, val_auc, *_ = evaluate(model, val_loader, device)
        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.6f} | "
            f"{time.time()-t0:.1f}s",
            flush=True,
        )

        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, best_path)
            print("  -> best saved:", best_path, flush=True)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, test_auc, y, s, b = evaluate(model, test_loader, device)
    print("\nTEST")
    print("loss:", test_loss)
    print("accuracy:", test_acc)
    print("AUC:", test_auc)
    for eff in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
        rej, fpr = rejection_at_eff(y, s, eff)
        print(f"Rej@{eff:.2f}: {rej}  FPR={fpr}")

    ev = out / "evaluation_test"
    ev.mkdir(exist_ok=True)
    np.save(ev / "labels.npy", y)
    np.save(ev / "scores.npy", s)
    np.save(ev / "betas.npy", b)
    with (ev / "metrics.json").open("w") as f:
        json.dump({"test_loss": test_loss, "accuracy": test_acc, "auc": test_auc}, f, indent=2)
    print("saved:", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dataset-tag", default="tof_mlp")
    p.add_argument("--mode", choices=("primary", "paddles", "all", "primary_beta"), default="primary")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--max-train-events", type=int)
    p.add_argument("--max-val-events", type=int)
    p.add_argument("--max-test-events", type=int)
    p.add_argument("--max-train-batches", type=int)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

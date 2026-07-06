#!/usr/bin/env python3
"""Train a small DGCNN-style GNN on Nakagami old VolID graph caches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EdgeConv, global_max_pool, global_mean_pool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output-dir", default="results/nakagami_old_volid_dgcnn_100k", type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature-mode",
        default="full",
        choices=["full", "edep_only", "no_edep", "pos_only", "edep_pos"],
        help="Ablation for node inputs. full uses x+pos; edep_only uses only log hit energy; "
        "no_edep drops log hit energy; pos_only uses only positions; edep_pos uses log hit energy + positions.",
    )
    return p.parse_args()


def sanitize_graph(g: Data) -> Data:
    # Drop string attrs because PyG batching of arbitrary metadata is not needed for training.
    return Data(
        x=g.x,
        pos=g.pos,
        edge_index=g.edge_index,
        volume_id=g.volume_id,
        beta=g.beta,
        y=g.y.view(1) if g.y.dim() == 0 else g.y,
    )


def load_split(data_dir: Path, split: str) -> list[Data]:
    path = data_dir / f"{split}.pt"
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    return [sanitize_graph(g) for g in graphs]


class SmallDGCNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, feature_mode: str):
        super().__init__()
        self.feature_mode = feature_mode
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ),
            aggr="max",
        )
        self.conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ),
            aggr="max",
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    def node_features(self, data: Data) -> torch.Tensor:
        if self.feature_mode == "full":
            return torch.cat([data.x, data.pos], dim=1)
        if self.feature_mode == "edep_only":
            return data.x[:, :1]
        if self.feature_mode == "no_edep":
            return torch.cat([data.x[:, 1:], data.pos], dim=1)
        if self.feature_mode == "pos_only":
            return data.pos
        if self.feature_mode == "edep_pos":
            return torch.cat([data.x[:, :1], data.pos], dim=1)
        raise ValueError(f"unknown feature_mode: {self.feature_mode}")

    def forward(self, data: Data) -> torch.Tensor:
        node = self.node_features(data)
        h = self.input(node)
        h = self.conv1(h, data.edge_index)
        h = self.conv2(h, data.edge_index)
        pooled = torch.cat(
            [
                global_mean_pool(h, data.batch),
                global_max_pool(h, data.batch),
            ],
            dim=1,
        )
        return self.head(pooled).view(-1)


def rejection_at(labels: np.ndarray, scores: np.ndarray, targets: list[float]) -> list[dict[str, float]]:
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    out = []
    for target in targets:
        idxs = np.flatnonzero(tpr >= target)
        idx = idxs[np.argmin(fpr[idxs])]
        rej = float("inf") if fpr[idx] == 0 else float(1.0 / fpr[idx])
        out.append(
            {
                "target_efficiency": target,
                "actual_efficiency": float(tpr[idx]),
                "fpr": float(fpr[idx]),
                "rejection": rej,
                "threshold": float(thr[idx]),
            }
        )
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, object]:
    model.eval()
    labels_all = []
    scores_all = []
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(batch)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch.y.detach().cpu().numpy()
        scores_all.append(scores)
        labels_all.append(labels)

    labels = np.concatenate(labels_all).astype(np.int64)
    scores = np.concatenate(scores_all)
    preds = (scores >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auc": float(roc_auc_score(labels, scores)),
        "rejection": rejection_at(labels, scores, [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]),
        "labels": labels,
        "scores": scores,
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_graphs = load_split(args.data_dir, "train")
    val_graphs = load_split(args.data_dir, "val")
    test_graphs = load_split(args.data_dir, "test")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    probe = Data(x=train_graphs[0].x, pos=train_graphs[0].pos)
    if args.feature_mode == "full":
        in_dim = int(probe.x.size(1) + probe.pos.size(1))
    elif args.feature_mode == "edep_only":
        in_dim = 1
    elif args.feature_mode == "no_edep":
        in_dim = int(probe.x.size(1) - 1 + probe.pos.size(1))
    elif args.feature_mode == "pos_only":
        in_dim = int(probe.pos.size(1))
    elif args.feature_mode == "edep_pos":
        in_dim = int(1 + probe.pos.size(1))
    else:
        raise ValueError(args.feature_mode)

    model = SmallDGCNN(in_dim=in_dim, hidden=args.hidden, feature_mode=args.feature_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print("device    :", device)
    print("data dir  :", args.data_dir)
    print("feature   :", args.feature_mode, f"(in_dim={in_dim})")
    print("train/val/test:", len(train_graphs), len(val_graphs), len(test_graphs))
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("AMP       :", args.amp and device.type == "cuda")

    best_auc = -1.0
    best_path = args.output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        correct = 0
        total = 0
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.y.float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(batch)
                loss = loss_fn(logits, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            pred = (torch.sigmoid(logits) >= 0.5).long()
            correct += int((pred.cpu() == batch.y.cpu()).sum())
            total += int(batch.y.numel())

        val = evaluate(model, val_loader, device, args.amp)
        train_acc = correct / max(total, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss: {np.mean(losses):.4f} train_acc: {train_acc:.4f} | "
            f"val_acc: {val['accuracy']:.4f} val_auc: {val['auc']:.5f} | "
            f"{elapsed:.1f}s"
        )

        if float(val["auc"]) > best_auc:
            best_auc = float(val["auc"])
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "best_auc": best_auc}, best_path)
            print("  -> best saved:", best_path)

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test = evaluate(model, test_loader, device, args.amp)

    np.save(args.output_dir / "labels.npy", test["labels"])
    np.save(args.output_dir / "scores.npy", test["scores"])
    metrics = {
        "accuracy": test["accuracy"],
        "auc": test["auc"],
        "rejection": test["rejection"],
        "best_epoch": int(state["epoch"]),
        "best_val_auc": float(state["best_auc"]),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\ntest accuracy:", f"{test['accuracy']:.6f}")
    print("test AUC     :", f"{test['auc']:.6f}")
    for r in test["rejection"]:
        print(f"Rej@{r['target_efficiency']:.2f}: {r['rejection']:.3f}  FPR={r['fpr']:.8g}")
    print("output:", args.output_dir)


if __name__ == "__main__":
    train(parse_args())

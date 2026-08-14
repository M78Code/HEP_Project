#!/usr/bin/env python3

"""
Sparse voxel GNN for TreeRec-derived voxel input.

Each nonzero Si(Li) voxel is treated as one graph node.
Node features are log1p(edep), normalized voxel indices, and occupancy.
TOF features are appended as graph-level features.  Legacy fixed-grid runs use
TOF paddle energy plus TOF primary features, while newer topIso fixed-grid
exports can use only the 11 primary TOF features because the paddle array is
known to be all zeros.
"""

import argparse, json, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool
from tqdm import tqdm

from GAPS_Project.src.models.gravnet import GravNetClassifier
from GAPS_Project.src.models.dgcnn import DGCNNClassifier

from sklearn.metrics import roc_auc_score, accuracy_score


def split_dir(data_dir, split):
    return Path(data_dir) / f"{split}_nakagami_style_4M"

class SparseVoxelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        split,
        max_events=None,
        k=8,
        tof_mean=None,
        tof_std=None,
        use_beta=False,
        tof_mode="paddles-primary",
    ):
        d = split_dir(data_dir, split)
        if not d.exists():
            raise FileNotFoundError(d)

        self.voxels = np.load(d / "voxels.npy", mmap_mode="r")
        self.tof_primary = np.load(d / "tof_primary.npy", mmap_mode="r")
        self.labels = np.load(d / "labels.npy", mmap_mode="r")
        self.betas = np.load(d / "betas.npy", mmap_mode="r")

        if tof_mode not in {"paddles-primary", "primary"}:
            raise ValueError(f"unknown tof_mode: {tof_mode}")
        self.tof_mode = tof_mode
        self.tof_paddles = None
        if self.tof_mode == "paddles-primary":
            paddle_path = d / "tof_paddles.npy"
            if not paddle_path.exists():
                raise FileNotFoundError(
                    f"{paddle_path} is required for tof_mode='paddles-primary'. "
                    "Use --tof-mode primary for topIso fixed-grid exports with zero paddles."
                )
            self.tof_paddles = np.load(paddle_path, mmap_mode="r")
        self.k = k
        self.tof_mean = None if tof_mean is None else np.asarray(tof_mean, dtype=np.float32)
        self.tof_std = None if tof_std is None else np.asarray(tof_std, dtype=np.float32)
        self.use_beta = bool(use_beta)

        n = len(self.labels)
        if max_events:
            labels_arr = np.asarray(self.labels)
            rng = np.random.default_rng(42)
            n0 = max_events // 2
            n1 = max_events - n0
            idx0 = np.flatnonzero(labels_arr == 0)
            idx1 = np.flatnonzero(labels_arr == 1)
            idx0 = rng.choice(idx0, size=min(n0, len(idx0)), replace=False)
            idx1 = rng.choice(idx1, size=min(n1, len(idx1)), replace=False)
            self.indices = np.concatenate([idx0, idx1])
            rng.shuffle(self.indices)
        else:
            self.indices = np.arange(n)
        self.n = len(self.indices)

    def __len__(self):
        return self.n

    def raw_tof(self, local_idx):
        idx = int(self.indices[int(local_idx)])
        if self.tof_mode == "primary":
            tof = self.tof_primary[idx].astype(np.float32)
        else:
            tof = np.concatenate(
                [
                    self.tof_paddles[idx].astype(np.float32),
                    self.tof_primary[idx].astype(np.float32),
                ]
            )
        if self.use_beta:
            tof = np.concatenate([tof, np.array([self.betas[idx]], dtype=np.float32)])
        return np.log1p(np.clip(tof, 0, None)).astype(np.float32)

    def _edge_index(self, pos):
        n = pos.shape[0]
        if n <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        k = min(self.k, n - 1)
        dist = torch.cdist(pos, pos)
        dist.fill_diagonal_(float("inf"))
        nn_idx = dist.topk(k, largest=False).indices

        src = torch.arange(n).repeat_interleave(k)
        dst = nn_idx.reshape(-1)
        return torch.stack([src, dst], dim=0).long()

    def __getitem__(self, idx):
        local_idx = int(idx)
        idx = int(self.indices[local_idx])
        v = self.voxels[idx]  # (10, 12, 12)
        z, x, y = np.nonzero(v > 0)

        if len(z) == 0:
            edep = np.array([0.0], dtype=np.float32)
            coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            occ = np.array([[0.0]], dtype=np.float32)
        else:
            edep = v[z, x, y].astype(np.float32)
            coords = np.stack(
                [z / 9.0, x / 11.0, y / 11.0], axis=1
            ).astype(np.float32)
            occ = np.ones((len(edep), 1), dtype=np.float32)

        edep_feat = np.log1p(edep).reshape(-1, 1).astype(np.float32)
        node_x = np.concatenate([edep_feat, coords, occ], axis=1)

        x_t = torch.from_numpy(node_x)
        pos_t = torch.from_numpy(coords)
        edge_index = self._edge_index(pos_t)

        tof = self.raw_tof(local_idx)
        if self.tof_mean is not None and self.tof_std is not None:
            tof = (tof - self.tof_mean) / self.tof_std

        return Data(
            x=x_t,
            pos=pos_t,
            edge_index=edge_index,
            y=torch.tensor([float(self.labels[idx])], dtype=torch.float32),
            tof=torch.from_numpy(tof).view(1, -1),
            beta=torch.tensor([float(self.betas[idx])], dtype=torch.float32),
        )


def compute_tof_standardizer(dataset, max_samples=None):
    n = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    if n <= 0:
        raise ValueError("cannot compute standardizer from empty dataset")

    mean = None
    m2 = None
    count = 0
    for i in range(n):
        x = dataset.raw_tof(i).astype(np.float64)
        if mean is None:
            mean = np.zeros_like(x, dtype=np.float64)
            m2 = np.zeros_like(x, dtype=np.float64)
        count += 1
        delta = x - mean
        mean += delta / count
        m2 += delta * (x - mean)

    var = m2 / max(count - 1, 1)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class SparseVoxelGNN(nn.Module):
    def __init__(self, hidden=128, tof_dim=183, dropout=0.15):
        super().__init__()
        self.conv1 = GraphConv(5, hidden)
        self.conv2 = GraphConv(hidden, hidden)
        self.conv3 = GraphConv(hidden, hidden)

        self.tof_mlp = nn.Sequential(
            nn.Linear(tof_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.cls = nn.Sequential(
            nn.Linear(hidden * 2 + 64, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        g = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=1,
        )

        tof = data.tof
        if tof.dim() == 1:
            tof = tof.view(g.size(0), -1)
        t = self.tof_mlp(tof)

        return self.cls(torch.cat([g, t], dim=1)).view(-1)


class SparseVoxelGravNet(nn.Module):
    def __init__(
        self,
        hidden=128,
        tof_dim=183,
        dropout=0.15,
        k=8,
        num_blocks=6,
    ):
        super().__init__()
        self.core = GravNetClassifier(
            in_channels=5,
            hidden_dim=hidden,
            k=k,
            num_classes=2,
            dropout=dropout,
            graph_feat_dim=tof_dim,
            num_blocks=num_blocks,
        )

    def forward(self, data):
        logits = self.core(
            data.x,
            data.edge_index,
            data.batch,
            graph_feat=data.tof,
        )
        return logits[:, 1] - logits[:, 0]


class SparseVoxelDGCNN(nn.Module):
    def __init__(
        self,
        hidden=64,
        tof_dim=183,
        dropout=0.3,
        k=8,
    ):
        super().__init__()
        self.core = DGCNNClassifier(
            in_channels=5,
            hidden_dim=hidden,
            k=k,
            num_classes=2,
            dropout=dropout,
            graph_feat_dim=tof_dim,
        )

    def forward(self, data):
        logits = self.core(
            data.x,
            data.edge_index,
            data.batch,
            graph_feat=data.tof,
        )
        return logits[:, 1] - logits[:, 0]


def rejection_at_eff(labels, scores, eff):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    sig = scores[labels == 1]
    bg = scores[labels == 0]
    if len(sig) == 0 or len(bg) == 0:
        return float("nan"), float("nan")

    thr = np.quantile(sig, 1.0 - eff)
    fpr = np.mean(bg >= thr)
    if fpr == 0:
        return float("inf"), 0.0
    return 1.0 / fpr, fpr


@torch.no_grad()
def evaluate(model, loader, device, desc="eval"):
    model.eval()
    ys, ss, bs = [], [], []
    total_loss, total_n = 0.0, 0

    for data in tqdm(loader, total=len(loader), desc=desc, leave=False, dynamic_ncols=True):
        data = data.to(device)
        logits = model(data)
        y = data.y.view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        prob = torch.sigmoid(logits)
        ys.append(y.cpu().numpy())
        ss.append(prob.cpu().numpy())
        bs.append(data.beta.view(-1).cpu().numpy())

        total_loss += float(loss.item()) * y.numel()
        total_n += y.numel()

    y = np.concatenate(ys)
    s = np.concatenate(ss)
    b = np.concatenate(bs)

    acc = accuracy_score(y.astype(int), (s >= 0.5).astype(int))
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
    return total_loss / max(total_n, 1), acc, auc, y, s, b


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path("results") / f"{datetime.now():%Y%m%d-%H%M%S}_SparseVoxelGNN_{args.dataset_tag}"
    out.mkdir(parents=True, exist_ok=True)

    train_base = SparseVoxelDataset(
        args.data_dir,
        "train",
        args.max_train_events,
        args.k,
        use_beta=args.use_beta,
        tof_mode=args.tof_mode,
    )
    tof_mean, tof_std = compute_tof_standardizer(train_base, args.standardize_samples)

    train_ds = SparseVoxelDataset(
        args.data_dir,
        "train",
        args.max_train_events,
        args.k,
        tof_mean,
        tof_std,
        args.use_beta,
        args.tof_mode,
    )
    val_ds = SparseVoxelDataset(
        args.data_dir,
        "val",
        args.max_val_events,
        args.k,
        tof_mean,
        tof_std,
        args.use_beta,
        args.tof_mode,
    )
    test_ds = SparseVoxelDataset(
        args.data_dir,
        "test",
        args.max_test_events,
        args.k,
        tof_mean,
        tof_std,
        args.use_beta,
        args.tof_mode,
    )

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    if args.num_workers > 0:
        loader_kw.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(train_ds, shuffle=not args.no_shuffle_train, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    tof_dim = int(train_ds.raw_tof(0).shape[0])
    if args.model == "graphconv":
        model = SparseVoxelGNN(
            hidden=args.hidden,
            tof_dim=tof_dim,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "gravnet":
        model = SparseVoxelGravNet(
            hidden=args.hidden,
            tof_dim=tof_dim,
            dropout=args.dropout,
            k=args.k,
            num_blocks=args.num_blocks,
        ).to(device)
    elif args.model == "dgcnn":
        model = SparseVoxelDGCNN(
            hidden=args.hidden,
            tof_dim=tof_dim,
            dropout=args.dropout,
            k=args.k,
        ).to(device)
    else:
        raise ValueError(f"unknown model: {args.model}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print("device:", device)
    print("train/val/test:", len(train_ds), len(val_ds), len(test_ds))
    print("batches:", len(train_loader), len(val_loader), len(test_loader))
    print("shuffle train:", not args.no_shuffle_train)
    print("use beta:", args.use_beta)
    print("tof mode:", args.tof_mode)
    print("tof/global dim:", tof_dim)
    print("model:", args.model)
    print("model params:", sum(p.numel() for p in model.parameters()))
    if args.early_stopping_patience > 0:
        print(
            "early stopping: "
            f"monitor=val_auc mode=max min_epochs={args.min_epochs} "
            f"patience={args.early_stopping_patience} "
            f"min_delta={args.early_stopping_min_delta}",
            flush=True,
        )

    best_val_auc = -float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    best_path = out / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, total_correct, total_n = 0.0, 0, 0

        train_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch:3d}/{args.epochs} [train]",
            leave=False,
            dynamic_ncols=True,
        )
        for bi, data in enumerate(train_bar):
            if args.max_train_batches and bi >= args.max_train_batches:
                break

            data = data.to(device)
            y = data.y.view(-1)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(data)
                loss = F.binary_cross_entropy_with_logits(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += int((pred == y).sum().item())
            total_loss += float(loss.item()) * y.numel()
            total_n += y.numel()
            train_bar.set_postfix(loss=f"{float(loss.item()):.4f}")

        val_loss, val_acc, val_auc, *_ = evaluate(
            model, val_loader, device, desc=f"Epoch {epoch:3d}/{args.epochs} [val]"
        )
        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.6f} | "
            f"{time.time()-t0:.1f}s",
            flush=True,
        )

        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "best_epoch": best_epoch,
            },
            out / "last.pt",
        )

        improved = np.isfinite(val_auc) and (
            val_auc > best_val_auc + args.early_stopping_min_delta
        )
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "tof_standardizer": {"mean": tof_mean, "std": tof_std},
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
            print(f"  -> best saved: {best_path} (val_auc={best_val_auc:.6f})", flush=True)
        else:
            no_improve_epochs += 1

        if (
            args.early_stopping_patience > 0
            and epoch >= args.min_epochs
            and no_improve_epochs >= args.early_stopping_patience
        ):
            print(
                "early stopping triggered: "
                f"epoch={epoch}, best_epoch={best_epoch}, "
                f"best_val_auc={best_val_auc:.6f}, "
                f"no_improve_epochs={no_improve_epochs}",
                flush=True,
            )
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    test_loss, test_acc, test_auc, y, s, b = evaluate(model, test_loader, device, desc="test")
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
    with open(ev / "metrics.json", "w") as f:
        json.dump({"test_loss": test_loss, "accuracy": test_acc, "auc": test_auc}, f, indent=2)

    print("saved:", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dataset-tag", default="voxel_sparse_gnn")
    p.add_argument("--model", choices=["graphconv", "gravnet", "dgcnn"], default="graphconv")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--max-train-events", type=int)
    p.add_argument("--max-val-events", type=int)
    p.add_argument("--max-test-events", type=int)
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--standardize-samples", type=int, default=None)
    p.add_argument("--use-beta", action="store_true")
    p.add_argument(
        "--no-shuffle-train",
        action="store_true",
        help=(
            "Disable DataLoader shuffling. Use this for large interleaved memmap "
            "datasets on NFS to avoid slow random reads."
        ),
    )
    p.add_argument(
        "--tof-mode",
        choices=["paddles-primary", "primary"],
        default="paddles-primary",
        help=(
            "Graph-level TOF input. 'paddles-primary' uses legacy 172 paddle + "
            "11 primary features; 'primary' uses only the 11 primary features."
        ),
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop after this many epochs without val_auc improvement. 0 disables early stopping.",
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=20,
        help="Minimum epochs before early stopping can trigger.",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-5,
        help="Minimum val_auc improvement required to reset early stopping patience.",
    )
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

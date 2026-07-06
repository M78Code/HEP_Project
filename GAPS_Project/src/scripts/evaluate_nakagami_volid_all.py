#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train_nakagami_volid_dgcnn import SmallDGCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def sanitize(g):
    return Data(
        x=g.x,
        pos=g.pos,
        edge_index=g.edge_index,
        volume_id=g.volume_id,
        beta=g.beta,
        y=g.y.view(1) if g.y.dim() == 0 else g.y,
    )


def load_graphs(data_dir):
    graphs = []
    meta = []
    for split in ["train", "val", "test"]:
        raw = torch.load(data_dir / f"{split}.pt",
                         map_location="cpu",
                         weights_only=False)
        for i, g in enumerate(raw):
            graphs.append(sanitize(g))
            meta.append({
                "split": split,
                "index": i,
                "file_id": getattr(g, "file_id", ""),
                "event_id": getattr(g, "event_id", ""),
                "beta": float(g.beta),
                "label": int(g.y),
            })
    return graphs, meta


def rejection_at(labels, scores, targets):
    fpr, tpr, thr = roc_curve(labels, scores,
                              drop_intermediate=False)
    out = []
    for target in targets:
        idxs = np.flatnonzero(tpr >= target)
        idx = idxs[np.argmin(fpr[idxs])]
        rej = float("inf") if fpr[idx] == 0 else float(1.0 / fpr[idx])
        out.append({
            "target_efficiency": target,
            "actual_efficiency": float(tpr[idx]),
            "fpr": float(fpr[idx]),
            "rejection": rej,
            "threshold": float(thr[idx]),
        })
    return out


@torch.no_grad()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )

    ck = torch.load(args.model_path,
                    map_location=device,
                    weights_only=False)
    ck_args = ck["args"]
    feature_mode = ck_args.get("feature_mode", "full")
    hidden = int(ck_args.get("hidden", 64))

    graphs, meta = load_graphs(args.data_dir)
    probe = graphs[0]

    if feature_mode == "full":
        in_dim = probe.x.size(1) + probe.pos.size(1)
    elif feature_mode == "edep_only":
        in_dim = 1
    elif feature_mode == "no_edep":
        in_dim = probe.x.size(1) - 1 + probe.pos.size(1)
    elif feature_mode == "pos_only":
        in_dim = probe.pos.size(1)
    elif feature_mode == "edep_pos":
        in_dim = 1 + probe.pos.size(1)
    else:
        raise ValueError(feature_mode)

    model = SmallDGCNN(in_dim=in_dim,
                       hidden=hidden,
                       feature_mode=feature_mode).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    loader = DataLoader(graphs,
                        batch_size=args.batch_size,
                        shuffle=False)

    labels_all = []
    scores_all = []

    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda",
                                enabled=args.amp and device.type == "cuda"):
            logits = model(batch)
        scores = torch.sigmoid(logits).cpu().numpy()
        labels = batch.y.cpu().numpy()
        scores_all.append(scores)
        labels_all.append(labels)

    labels = np.concatenate(labels_all).astype(np.int64)
    scores = np.concatenate(scores_all)
    preds = (scores >= 0.5).astype(np.int64)

    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "scores.npy", scores)
    np.save(args.output_dir / "preds.npy", preds)
    np.save(args.output_dir / "betas.npy",
            np.array([m["beta"] for m in meta], dtype=np.float32))

    with open(args.output_dir / "predictions.csv",
              "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "split", "index", "file_id", "event_id",
            "beta", "label", "score", "pred"
        ])
        for m, score, pred in zip(meta, scores, preds):
            w.writerow([
                m["split"], m["index"],
                m["file_id"], m["event_id"],
                m["beta"], m["label"],
                float(score), int(pred)
            ])

    metrics = {
        "n_events": int(len(labels)),
        "label_counts": {
            "0": int((labels == 0).sum()),
            "1": int((labels == 1).sum()),
        },
        "accuracy": float(accuracy_score(labels, preds)),
        "auc": float(roc_auc_score(labels, scores)),
        "rejection": rejection_at(
            labels, scores, [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
        ),
        "feature_mode": feature_mode,
        "best_epoch": int(ck.get("epoch", -1)),
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("events:", len(labels))
    print("accuracy:", f"{metrics['accuracy']:.6f}")
    print("AUC:", f"{metrics['auc']:.6f}")
    for r in metrics["rejection"]:
        print(
            f"Rej@{r['target_efficiency']:.2f}: "
            f"{r['rejection']}  FPR={r['fpr']}"
        )
    print("output:", args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Fig.7.2-style CNN+DNN on exported Nakagami fixed-grid arrays.

This evaluator is paired with src/train/train_nakagami_cnndnn.py.  It loads a
saved best.pt/last.pt checkpoint and evaluates one split without running any
additional training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from GAPS_GNN_Study.src.train.train_nakagami_cnndnn import (
    NakagamiFig72CNNDNNDataset,
    infer,
    save_evaluation,
)
from GAPS_GNN_Study.src.models.cnn_dnn_hybrid import CNNDNNHybrid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    return p.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        ckpt_args = ckpt.get("args", {}) or {}
    else:
        state = ckpt
        ckpt_args = {}
    return state, ckpt_args, ckpt


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_enabled = args.amp and device.type == "cuda"

    state, ckpt_args, ckpt = load_checkpoint(args.model_path, device)
    dropout = args.dropout
    if dropout is None:
        dropout = float(ckpt_args.get("dropout", 0.3))

    dataset = NakagamiFig72CNNDNNDataset(args.data_dir, args.split, args.max_events)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = CNNDNNHybrid(tof_dim=11, dropout=dropout).to(device)
    model.load_state_dict(state)

    print(f"device     : {device}", flush=True)
    print(f"data dir   : {args.data_dir}", flush=True)
    print(f"split      : {args.split}", flush=True)
    print(f"events     : {len(dataset):,}", flush=True)
    print(f"batches    : {len(loader):,}", flush=True)
    print("input      : voxels.npy + tof_primary.npy (beta ignored)", flush=True)
    print("model      : CNNDNNHybrid / Nakagami Fig.7.2 style", flush=True)
    print(f"parameters : {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"dropout    : {dropout}", flush=True)
    print(f"AMP        : {amp_enabled} ({args.amp_dtype if amp_enabled else 'disabled'})", flush=True)
    print(f"checkpoint : {args.model_path}", flush=True)
    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print(f"checkpoint epoch: {ckpt['epoch']}", flush=True)
        if "best_val_loss" in ckpt:
            print(f"best_val_loss   : {ckpt['best_val_loss']}", flush=True)

    test_loss, labels, scores, betas = infer(model, loader, device, amp_enabled, amp_dtype)
    metrics = save_evaluation(args.output_dir, args.model_path, labels, scores, betas, test_loss)

    print("\nEVALUATION", flush=True)
    print(f"loss    : {metrics['test_loss']}", flush=True)
    print(f"accuracy: {metrics['accuracy']}", flush=True)
    print(f"AUC     : {metrics['auc']}", flush=True)
    for row in metrics["rejection"]:
        print(
            f"Rej@{row['target_efficiency']:.2f}: {row['rejection']}  FPR={row['fpr']}",
            flush=True,
        )
    print(f"saved: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

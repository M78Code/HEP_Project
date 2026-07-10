#!/usr/bin/env python3
"""Summarize beta distributions and simple feature separability for graph caches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--bins",
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated beta bin edges.",
    )
    p.add_argument("--title", default=None)
    return p.parse_args()


def load_graphs(data_dir: Path) -> list:
    graphs = []
    for split in ["train", "val", "test"]:
        graphs.extend(torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=False))
    return graphs


def oriented_auc(y: np.ndarray, x: np.ndarray) -> float:
    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def graph_features(graphs: list) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {
        "beta": [],
        "n_hits": [],
        "edep_sum": [],
        "edep_mean": [],
        "tof_mean": [],
        "stop_mean": [],
        "vol_mean": [],
        "pos_x_mean": [],
        "pos_y_mean": [],
        "pos_z_mean": [],
    }
    for g in graphs:
        x = g.x.detach().cpu()
        pos = g.pos.detach().cpu()
        edep = np.expm1(x[:, 0].numpy())
        out["beta"].append(float(g.beta))
        out["n_hits"].append(float(x.size(0)))
        out["edep_sum"].append(float(edep.sum()))
        out["edep_mean"].append(float(edep.mean()))
        out["tof_mean"].append(float(x[:, 1].mean()) if x.size(1) > 1 else float("nan"))
        out["stop_mean"].append(float(x[:, 2].mean()) if x.size(1) > 2 else float("nan"))
        out["vol_mean"].append(float(x[:, 3].mean()) if x.size(1) > 3 else float("nan"))
        out["pos_x_mean"].append(float(pos[:, 0].mean()))
        out["pos_y_mean"].append(float(pos[:, 1].mean()))
        out["pos_z_mean"].append(float(pos[:, 2].mean()))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def plot_beta(beta: np.ndarray, y: np.ndarray, bins: list[float], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.hist(
        beta[y == 0],
        bins=bins,
        histtype="step",
        linewidth=2.2,
        label="anti-proton",
        color="#1f77b4",
    )
    ax.hist(
        beta[y == 1],
        bins=bins,
        histtype="step",
        linewidth=2.2,
        label="anti-deuteron",
        color="#ff7f0e",
    )
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Events")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bins = [float(x) for x in args.bins.split(",")]

    graphs = load_graphs(args.data_dir)
    y = np.asarray([int(g.y) for g in graphs], dtype=np.int64)
    feats = graph_features(graphs)
    beta = feats["beta"]

    hist_rows = []
    print(f"events: {len(graphs):,}")
    print("labels:", np.bincount(y, minlength=2).tolist())
    print("beta bins:", bins)
    print(f'{"beta range":>14} {"antiP":>9} {"antiD":>9}')
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        is_last = i == len(bins) - 2
        mask = (beta >= low) & ((beta <= high) if is_last else (beta < high))
        anti_p = int(np.sum(mask & (y == 0)))
        anti_d = int(np.sum(mask & (y == 1)))
        row = {"beta_low": low, "beta_high": high, "antiP": anti_p, "antiD": anti_d}
        hist_rows.append(row)
        print(f'[{low:.2f},{high:.2f}{"]" if is_last else ")":>1} {anti_p:9,d} {anti_d:9,d}')

    feature_rows = []
    print(f'\n{"feature":>12} {"AUC":>9} {"antiP mean":>14} {"antiD mean":>14}')
    for name, values in feats.items():
        finite = np.isfinite(values)
        if len(np.unique(y[finite])) < 2:
            auc = float("nan")
        else:
            auc = oriented_auc(y[finite], values[finite])
        row = {
            "feature": name,
            "auc": auc,
            "antiP_mean": finite_mean(values[(y == 0) & finite]),
            "antiD_mean": finite_mean(values[(y == 1) & finite]),
        }
        feature_rows.append(row)
        print(f'{name:>12} {auc:9.5f} {row["antiP_mean"]:14.6g} {row["antiD_mean"]:14.6g}')

    title = args.title or args.data_dir.name
    plot_beta(beta, y, bins, args.out_dir / "beta_distribution.png", title)

    with open(args.out_dir / "beta_hist.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["beta_low", "beta_high", "antiP", "antiD"])
        writer.writeheader()
        writer.writerows(hist_rows)
    with open(args.out_dir / "feature_auc.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "auc", "antiP_mean", "antiD_mean"])
        writer.writeheader()
        writer.writerows(feature_rows)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": str(args.data_dir),
                "n_events": int(len(graphs)),
                "label_counts": np.bincount(y, minlength=2).astype(int).tolist(),
                "beta_bins": hist_rows,
                "feature_auc": feature_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("saved:", args.out_dir)


if __name__ == "__main__":
    main()

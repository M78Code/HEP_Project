#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


MODEL_SPECS = [
    ("CNN+DNN", "results/evaluation_aohba4M_nakagami3input_amp_bs256", None),
    ("GravNet", "results/evaluation_aohba4M_atrest_gravnet_mc1", None),
    ("GravNet+TOF", "results/evaluation_aohba4M_atrest_tof172_balanced", None),
    ("DGCNN", "results/evaluation_aohba4M_dgcnn", None),
    ("FusedGravNet", "results/evaluation_aohba4M_fused_amp", None),
]

DATA_AMOUNT_SPECS = [
    ("4M", "results/evaluation_aohba4M_atrest_gravnet_mc1", None),
    ("50M", "results/evaluation_aohba50M_gravnet_on_4Mtest", None),
    ("50M_latest", "results/evaluation_aohba50M_gravnet_on_4Mtest_latest_current", None),
]


def load_eval(repo: Path, directory: str, prefix: str | None):
    d = repo / directory
    if prefix is None:
        labels_path = d / "labels.npy"
        score_path = d / "scores.npy"
        if not score_path.exists():
            score_path = d / "probs.npy"
        beta_path = d / "betas.npy"
    else:
        labels_path = d / f"{prefix}_labels.npy"
        score_path = d / f"{prefix}_probs.npy"
        beta_path = d / f"{prefix}_betas.npy"

    if not labels_path.exists() or not score_path.exists():
        return None

    labels = np.load(labels_path)
    scores = np.load(score_path)
    betas = np.load(beta_path) if beta_path.exists() else None
    return labels.astype(int), scores.astype(float), betas


def rejection_at(labels: np.ndarray, scores: np.ndarray, target: float):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    idxs = np.flatnonzero(tpr >= target)
    if len(idxs) == 0:
        return math.nan, math.nan, math.nan
    idx = idxs[np.argmin(fpr[idxs])]
    rej = math.inf if fpr[idx] == 0 else 1.0 / fpr[idx]
    return float(rej), float(tpr[idx]), float(fpr[idx])


def metrics_for(labels: np.ndarray, scores: np.ndarray):
    out = {"auc": float(roc_auc_score(labels, scores))}
    for target in [0.50, 0.70, 0.80, 0.90, 0.98]:
        rej, eff, fpr = rejection_at(labels, scores, target)
        out[f"rej@{target:.2f}"] = rej
        out[f"eff@{target:.2f}"] = eff
        out[f"fpr@{target:.2f}"] = fpr
    return out


def parse_training_log(log_path: Path):
    # Matches both "Epoch   1/80 | train_loss: ... train_acc: ... | val_loss: ... val_acc: ..."
    pat = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+train_loss:\s+([0-9.]+)\s+train_acc:\s+([0-9.]+)\s+\|\s+val_loss:\s+([0-9.]+)\s+val_acc:\s+([0-9.]+)"
    )
    epochs, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if not m:
            continue
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        train_acc.append(float(m.group(3)))
        val_loss.append(float(m.group(4)))
        val_acc.append(float(m.group(5)))
    if not epochs:
        return None
    return {
        "epoch": np.array(epochs),
        "train_acc": np.array(train_acc),
        "val_acc": np.array(val_acc),
        "train_loss": np.array(train_loss),
        "val_loss": np.array(val_loss),
    }


def plot_learning_curve(log_data, out: Path):
    plt.figure(figsize=(4.2, 3.0), dpi=220)
    plt.plot(log_data["epoch"], log_data["train_acc"], marker="o", markersize=2.4, linewidth=1.4, label="Train")
    plt.plot(log_data["epoch"], log_data["val_acc"], marker="o", markersize=2.4, linewidth=1.4, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.84, min(1.0, max(log_data["train_acc"].max(), log_data["val_acc"].max()) + 0.015))
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_score_hist(labels, scores, out: Path):
    plt.figure(figsize=(4.2, 3.0), dpi=220)
    bins = np.linspace(0.0, 1.0, 101)
    plt.hist(scores[labels == 0], bins=bins, histtype="step", linewidth=1.4, label="anti-proton", color="#d95f02", log=True)
    plt.hist(scores[labels == 1], bins=bins, histtype="step", linewidth=1.4, label="anti-deuteron", color="#1b9e77", log=True)
    plt.xlabel("Classification score for anti-deuteron")
    plt.ylabel("Counts")
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_rejection_curve(evals, out: Path):
    plt.figure(figsize=(4.2, 3.0), dpi=220)
    for name, labels, scores in evals:
        fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=True)
        mask = fpr > 0
        plt.plot(tpr[mask], 1.0 / fpr[mask], linewidth=1.4, label=name)
    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.xlim(0.45, 1.0)
    plt.ylim(1, 1e5)
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--out-dir", type=Path, default=Path("yokou_ch4_outputs"))
    ap.add_argument("--gravnet-log", type=Path, default=Path.home() / "train_aohba4M_atrest_gravnet_mc1.log")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    loaded = {}
    for name, directory, prefix in MODEL_SPECS:
        data = load_eval(repo, directory, prefix)
        if data is None:
            print(f"[missing] {name}: {directory}")
            continue
        labels, scores, betas = data
        rows[name] = metrics_for(labels, scores)
        loaded[name] = (labels, scores, betas)

    (out_dir / "metrics_rej70.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\nTable 2 candidates")
    print("model, auc, rej50, rej70, rej90")
    for name in ["CNN+DNN", "GravNet", "GravNet+TOF", "DGCNN", "FusedGravNet"]:
        if name not in rows:
            continue
        r = rows[name]
        print(f"{name}, {r['auc']:.4f}, {r['rej@0.50']:.2f}, {r['rej@0.70']:.2f}, {r['rej@0.90']:.2f}")

    amount_rows = {}
    amount_loaded = {}
    for name, directory, prefix in DATA_AMOUNT_SPECS:
        data = load_eval(repo, directory, prefix)
        if data is None:
            print(f"[missing] {name}: {directory}")
            continue
        labels, scores, _ = data
        amount_rows[name] = metrics_for(labels, scores)
        amount_loaded[name] = (labels, scores)
    (out_dir / "metrics_data_amount_rej70.json").write_text(json.dumps(amount_rows, indent=2), encoding="utf-8")

    print("\nTable 3 candidates")
    print("data, auc, rej50, rej70, rej90")
    for name in ["4M", "50M", "50M_latest"]:
        if name not in amount_rows:
            continue
        r = amount_rows[name]
        print(f"{name}, {r['auc']:.4f}, {r['rej@0.50']:.2f}, {r['rej@0.70']:.2f}, {r['rej@0.90']:.2f}")

    if not args.skip_plots:
        log_data = parse_training_log(args.gravnet_log)
        if log_data is None:
            print(f"[missing] learning log: {args.gravnet_log}")
        else:
            plot_learning_curve(log_data, out_dir / "fig2_gravnet_learning_curve.png")
            print("saved", out_dir / "fig2_gravnet_learning_curve.png")

        if "GravNet" in loaded:
            labels, scores, _ = loaded["GravNet"]
            plot_score_hist(labels, scores, out_dir / "fig3_gravnet_score_distribution.png")
            print("saved", out_dir / "fig3_gravnet_score_distribution.png")
        else:
            print("[missing] GravNet eval for score histogram")

        evals = [(name, loaded[name][0], loaded[name][1]) for name in ["CNN+DNN", "GravNet", "GravNet+TOF", "DGCNN", "FusedGravNet"] if name in loaded]
        if "50M_latest" in amount_loaded:
            labels_50m, scores_50m = amount_loaded["50M_latest"]
            evals.append(("GravNet 50M", labels_50m, scores_50m))
        if len(evals) >= 2:
            plot_rejection_curve(evals, out_dir / "fig4_rejection_curve.png")
            print("saved", out_dir / "fig4_rejection_curve.png")
        else:
            print("[missing] not enough evals for rejection curve")


if __name__ == "__main__":
    main()

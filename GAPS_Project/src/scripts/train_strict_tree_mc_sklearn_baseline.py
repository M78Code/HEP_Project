#!/usr/bin/env python3
"""Ordinary-ML baseline on strict TreeMc primary-track graphs.

Purpose
-------
The strict VolID (TreeMc primary-track) graphs are near-perfectly separable by
GravNet/DGCNN using only per-hit energy loss (edep_only -> AUC ~ 1.0).  This
script tests whether that high performance is driven by the *input information*
rather than the graph model, by classifying the same events with plain
event-level summary features + scikit-learn classifiers (no GNN, no CNN).

It reads the same ``{train,val,test}.pt`` PyG caches produced by
``export_local430_strict_volid_graphs.py`` / ``export_nakagami_volid_graphs.py``.
Each graph is ``Data(x=[N,4], pos=[N,3], volume_id=[N], y, beta)`` with
``x[:,0]=log1p(edep), x[:,1]=tof/50, x[:,2]=stoplayer/10, x[:,3]=volume_id/1e8``.

Feature modes (mirror the GNN node-feature ablation):
  edep     : energy-loss summary statistics only (no geometry, no n_hits)
  edep_geo : edep stats + n_hits + hit-position summary
  full     : edep_geo + tof/stoplayer/unique-volume summary

Outputs (per --out-dir):
  metrics.json                 all (mode, model) AUC + Rej@eff
  labels.npy                   test labels (shared)
  betas.npy                    test betas (for optional beta-binned analysis)
  scores_<mode>_<model>.npy    test scores per configuration

The metric definitions (roc_curve + Rej = 1/FPR at fixed efficiency) match
``make_yokou_ch4_figures.py`` and ``make_selection_ablation_summary.py`` so the
numbers are directly comparable with the GNN results.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Dir containing train.pt / val.pt / test.pt")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--feature-modes", default="edep,edep_geo,full")
    p.add_argument("--models", default="hgb,logreg")
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(data_dir: Path, split: str):
    graphs = torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=False)
    labels = np.array([int(g.y) for g in graphs], dtype=np.int64)
    betas = np.array([float(g.beta) for g in graphs], dtype=np.float32)
    return graphs, labels, betas


def _edep_stats(edep: np.ndarray) -> list[float]:
    q = np.quantile(edep, QUANTILES)
    return [
        float(edep.sum()), float(edep.mean()), float(edep.std()),
        float(edep.min()), float(edep.max()), *[float(v) for v in q],
    ]


def event_features(g, mode: str) -> list[float]:
    x = g.x.numpy()
    pos = g.pos.numpy()
    edep = x[:, 0]            # log1p(edep) per hit
    n = int(edep.shape[0])
    feats = _edep_stats(edep)
    if mode == "edep":
        return feats

    feats.append(float(n))   # n_hits
    feats += [
        float(pos[:, 0].mean()), float(pos[:, 1].mean()), float(pos[:, 2].mean()),
        float(pos[:, 0].std()), float(pos[:, 1].std()), float(pos[:, 2].std()),
        float(pos[:, 2].max() - pos[:, 2].min()),                 # z extent (range proxy)
        float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))),  # bbox diagonal
    ]
    if mode == "edep_geo":
        return feats

    tof = x[:, 1]
    stoplayer = x[:, 2]
    volid = g.volume_id.numpy()
    feats += [
        float(tof.mean()), float(tof.std()), float(tof.max()),
        float(stoplayer.mean()), float(stoplayer.max()),
        float(np.unique(volid).size),
    ]
    if mode == "full":
        return feats
    raise ValueError(f"unknown feature mode: {mode}")


def build_matrix(graphs, mode: str) -> np.ndarray:
    mat = np.array([event_features(g, mode) for g in graphs], dtype=np.float64)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def rejection_at(labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    out = []
    for t in TARGETS:
        idxs = np.flatnonzero(tpr >= t)
        if len(idxs) == 0:
            out.append({"target": t, "rejection": math.nan, "fpr": math.nan, "eff": math.nan})
            continue
        idx = idxs[np.argmin(fpr[idxs])]
        rej = math.inf if fpr[idx] == 0 else 1.0 / float(fpr[idx])
        out.append({"target": t, "rejection": rej, "fpr": float(fpr[idx]), "eff": float(tpr[idx])})
    return out


def make_model(name: str, seed: int):
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, early_stopping=True,
            validation_fraction=0.1, random_state=seed,
        )
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown model: {name}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.feature_modes.split(",") if m.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"loading {args.data_dir} ...", flush=True)
    train_graphs, y_train, _ = load_split(args.data_dir, args.train_split)
    test_graphs, y_test, beta_test = load_split(args.data_dir, args.test_split)
    print(f"train={len(train_graphs):,}  test={len(test_graphs):,}  "
          f"test labels={np.bincount(y_test).tolist()}", flush=True)

    np.save(args.out_dir / "labels.npy", y_test)
    np.save(args.out_dir / "betas.npy", beta_test)

    results = []
    print("\nmode, model, auc, rej@0.50, rej@0.70, rej@0.90", flush=True)
    for mode in modes:
        Xtr = build_matrix(train_graphs, mode)
        Xte = build_matrix(test_graphs, mode)
        for name in models:
            t0 = time.time()
            model = make_model(name, args.seed)
            model.fit(Xtr, y_train)
            scores = model.predict_proba(Xte)[:, 1]
            np.save(args.out_dir / f"scores_{mode}_{name}.npy", scores.astype(np.float32))
            auc = float(roc_auc_score(y_test, scores))
            rej = rejection_at(y_test, scores)
            r = {tt["target"]: tt["rejection"] for tt in rej}
            results.append({
                "feature_mode": mode, "model": name, "n_features": int(Xtr.shape[1]),
                "auc": auc, "rejection": rej, "elapsed_sec": round(time.time() - t0, 1),
            })
            print(f"{mode}, {name}, {auc:.6f}, "
                  f"{r[0.50]:.1f}, {r[0.70]:.1f}, {r[0.90]:.1f}", flush=True)

    (args.out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, default=lambda o: "inf" if o == math.inf else o),
        encoding="utf-8",
    )
    print(f"\nsaved -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ordinary-ML baseline on GAPS graphs (TreeMc strict / TreeRec) with sklearn.

Purpose
-------
The strict VolID (TreeMc primary-track) graphs are near-perfectly separable by
GravNet/DGCNN using only per-hit energy loss.  This script tests whether that
high performance is driven by the *input information* rather than the graph
model, by classifying the same events with plain event-level summary features
+ scikit-learn classifiers (no GNN, no CNN).

Two input layouts:

  --layout volid  (default)
    Reads {train,val,test}.pt from export_local430_strict_volid_graphs.py etc.
    Node x=[N,4] with x[:,0]=log1p(edep), x[:,1]=tof/50, x[:,2]=stoplayer/10,
    x[:,3]=volume_id/1e8; pos=[N,3]; volume_id=[N]; beta.

  --layout treerec
    Reads sharded {split}_mixed_*.pt from the TreeRec mixed graph cache.
    Node x=[N,8]=[x,y,z,energy,time,dE/dx,det_type,layer_norm]; graph_feat=[45];
    mc_beta.  Used as the ML-2 contrast (realistic reconstructed input).

Feature modes:
  n_hits    hit multiplicity only
  edep_sum  total energy loss only
  edep_mean mean energy loss only
  edep_quantile energy-loss quantiles only
  edep_no_max energy-loss shape statistics without sum/max
  stoplayer volid only: stopping-layer summary only
  tof_only  volid only: TOF-like summary only
  vol_count volid only: number of unique volume IDs only
  edep_shape energy-loss shape statistics only (no sum -> multiplicity-independent)
  edep      energy-loss summary statistics only (includes sum)
  edep_geo  edep stats + n_hits + hit-position summary
  full      volid: edep_geo + tof/stoplayer/unique-volume ;
            treerec: edep_geo + 45-D graph_feat
  graphfeat treerec only: the 45-D hand-crafted graph_feat vector alone

Outputs (per --out-dir): metrics.json, labels.npy, betas.npy,
scores_<mode>_<model>.npy.  Metric definitions (roc_curve + Rej = 1/FPR at
fixed efficiency) match make_yokou_ch4_figures.py so numbers are directly
comparable with the GNN results.
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
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--layout", choices=["volid", "treerec"], default="volid")
    p.add_argument("--feature-modes", default="edep,edep_geo,full")
    p.add_argument("--models", default="hgb,logreg")
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_split_files(data_dir: Path, split: str) -> list[Path]:
    single = data_dir / f"{split}.pt"
    if single.exists():
        return [single]
    shards = sorted(data_dir.glob(f"{split}_mixed_*.pt"))
    if shards:
        return shards
    manifest_path = data_dir / "subset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_dir = Path(manifest["source_dir"])
        if not source_dir.exists():
            candidates = [
                data_dir.parent / source_dir.name,
                data_dir.parent / "aohba_atrest_tof172_sharded",
            ]
            source_dir = next((p for p in candidates if p.exists()), source_dir)

        split_info = manifest["splits"][split]
        files = []
        for particle in sorted(split_info["particles"]):
            files.extend(source_dir / name for name in split_info["particles"][particle]["files"])

        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest shard files are missing; first missing: {missing[0]}"
            )
        return files
    raise FileNotFoundError(f"no {split}.pt or {split}_mixed_*.pt in {data_dir}")


def extract_record(g, layout: str) -> dict:
    x = g.x.numpy()
    if layout == "volid":
        return {
            "edep": x[:, 0], "pos": g.pos.numpy(),
            "tof": x[:, 1], "stoplayer": x[:, 2], "volid": g.volume_id.numpy(),
        }
    if hasattr(g, "graph_feat"):
        gf = g.graph_feat.numpy().reshape(-1)
    else:
        parts = []
        for attr in ("n_hits", "total_energy", "sili_profile", "tof_profile", "tof_feat"):
            if hasattr(g, attr):
                parts.append(torch.as_tensor(getattr(g, attr)).detach().cpu().numpy().reshape(-1))
        gf = np.concatenate(parts).astype(np.float32) if parts else None
    return {"edep": np.log1p(np.clip(x[:, 3], 0.0, None)), "pos": x[:, :3], "graph_feat": gf}


def graph_label_beta(g, layout: str) -> tuple[int, float]:
    label = int(g.y.view(()).item()) if g.y.dim() > 0 else int(g.y)
    if hasattr(g, "beta"):
        beta = float(g.beta.view(()).item()) if torch.is_tensor(g.beta) else float(g.beta)
    elif hasattr(g, "mc_beta"):
        beta = float(g.mc_beta.view(()).item())
    else:
        beta = -1.0
    return label, beta


def load_split(files: list[Path], layout: str):
    recs, labels, betas = [], [], []
    for f in files:
        graphs = torch.load(f, map_location="cpu", weights_only=False)
        for g in graphs:
            recs.append(extract_record(g, layout))
            lab, bta = graph_label_beta(g, layout)
            labels.append(lab)
            betas.append(bta)
        del graphs
        print(f"  loaded {f.name}: total events={len(recs):,}", flush=True)
    return recs, np.array(labels, dtype=np.int64), np.array(betas, dtype=np.float32)


def _edep_stats(edep: np.ndarray) -> list[float]:
    q = np.quantile(edep, QUANTILES)
    return [float(edep.sum()), float(edep.mean()), float(edep.std()),
            float(edep.min()), float(edep.max()), *[float(v) for v in q]]


def _scalar_stats(xs: np.ndarray) -> list[float]:
    q = np.quantile(xs, QUANTILES)
    return [float(xs.mean()), float(xs.std()), float(xs.min()), float(xs.max()),
            *[float(v) for v in q]]


def _pos_stats(pos: np.ndarray) -> list[float]:
    return [
        float(pos[:, 0].mean()), float(pos[:, 1].mean()), float(pos[:, 2].mean()),
        float(pos[:, 0].std()), float(pos[:, 1].std()), float(pos[:, 2].std()),
        float(pos[:, 2].max() - pos[:, 2].min()),
        float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))),
    ]


def event_features(rec: dict, mode: str, layout: str) -> list[float]:
    if mode == "graphfeat":
        if rec.get("graph_feat") is None:
            raise ValueError("graphfeat mode requires treerec layout with graph_feat")
        return [float(v) for v in rec["graph_feat"]]

    feats = _edep_stats(rec["edep"])
    if mode == "n_hits":
        return [float(rec["edep"].shape[0])]
    if mode == "edep_sum":
        return [feats[0]]
    if mode == "edep_mean":
        return [feats[1]]
    if mode == "edep_quantile":
        return feats[5:]
    if mode == "edep_no_max":
        return [feats[1], feats[2], feats[3], *feats[5:]]
    if mode == "stoplayer":
        if layout != "volid":
            raise ValueError("stoplayer mode requires volid layout")
        return _scalar_stats(rec["stoplayer"])
    if mode == "tof_only":
        if layout != "volid":
            raise ValueError("tof_only mode requires volid layout")
        return _scalar_stats(rec["tof"])
    if mode == "vol_count":
        if layout != "volid":
            raise ValueError("vol_count mode requires volid layout")
        return [float(np.unique(rec["volid"]).size)]
    if mode == "edep":
        return feats
    if mode == "edep_shape":
        return feats[1:]  # drop sum -> multiplicity-independent energy shape stats
    feats.append(float(rec["edep"].shape[0]))          # n_hits
    feats += _pos_stats(rec["pos"])
    if mode == "edep_geo":
        return feats
    if mode == "full":
        if layout == "volid":
            feats += [
                float(rec["tof"].mean()), float(rec["tof"].std()), float(rec["tof"].max()),
                float(rec["stoplayer"].mean()), float(rec["stoplayer"].max()),
                float(np.unique(rec["volid"]).size),
            ]
        else:
            feats += [float(v) for v in rec["graph_feat"]]
        return feats
    raise ValueError(f"unknown feature mode: {mode}")


def build_matrix(recs: list[dict], mode: str, layout: str) -> np.ndarray:
    mat = np.array([event_features(r, mode, layout) for r in recs], dtype=np.float64)
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
            validation_fraction=0.1, random_state=seed)
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

    print(f"loading train ({args.layout}) ...", flush=True)
    train_recs, y_train, _ = load_split(resolve_split_files(args.data_dir, args.train_split), args.layout)
    print(f"loading test ({args.layout}) ...", flush=True)
    test_recs, y_test, beta_test = load_split(resolve_split_files(args.data_dir, args.test_split), args.layout)
    print(f"train={len(train_recs):,}  test={len(test_recs):,}  "
          f"test labels={np.bincount(y_test).tolist()}", flush=True)

    np.save(args.out_dir / "labels.npy", y_test)
    np.save(args.out_dir / "betas.npy", beta_test)

    results = []
    print("\nmode, model, auc, rej@0.50, rej@0.70, rej@0.90", flush=True)
    for mode in modes:
        print(f"[{time.strftime('%H:%M:%S')}] extracting: mode={mode} ...", flush=True)
        Xtr = build_matrix(train_recs, mode, args.layout)
        Xte = build_matrix(test_recs, mode, args.layout)
        for name in models:
            t0 = time.time()
            model = make_model(name, args.seed)
            model.fit(Xtr, y_train)
            scores = model.predict_proba(Xte)[:, 1]
            np.save(args.out_dir / f"scores_{mode}_{name}.npy", scores.astype(np.float32))
            auc = float(roc_auc_score(y_test, scores))
            rej = rejection_at(y_test, scores)
            r = {tt["target"]: tt["rejection"] for tt in rej}
            results.append({"feature_mode": mode, "model": name, "layout": args.layout,
                            "n_features": int(Xtr.shape[1]), "auc": auc,
                            "rejection": rej, "elapsed_sec": round(time.time() - t0, 1)})
            print(f"{mode}, {name}, {auc:.6f}, {r[0.50]:.1f}, {r[0.70]:.1f}, {r[0.90]:.1f}", flush=True)

    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

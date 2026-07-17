#!/usr/bin/env python3
"""Fast sklearn baselines for Nakagami fixed-length Atrest CSV.

This script checks whether the fixed-length Atrest CSV itself contains strong
classification information before spending time on GNN/CNN training.
It is intentionally simple and supports the currently confirmed 1457-column
layout:

  col 0       : random seed / file id
  col 1       : entry index
  col 2       : label (0=pbar, 1=dbar)
  col 4       : beta
  col 6:1446  : Si(Li) voxel energy, 1440 values
  col 1446:1457: 11 TOF/global features
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


N_VOXEL = 10 * 12 * 12


def infer_particle_label(path: Path) -> int | None:
    name = path.name.lower()
    if "pbar" in name or "antip" in name:
        return 0
    if "dbar" in name or "antid" in name:
        return 1
    return None


def list_files(inputs: list[str], pattern: str) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob(pattern)))
        elif p.is_file():
            files.append(p)
        else:
            files.extend(Path(x) for x in sorted(glob.glob(item)))
    return [p for p in files if p.suffix == ".csv" and "VolID" not in p.name]


def parse_row(row: list[str]) -> tuple[np.ndarray, np.ndarray, int, float]:
    if len(row) != 1457:
        raise ValueError(f"expected 1457 columns, got {len(row)}")

    label = int(float(row[2]))
    beta = float(row[4])
    si = np.asarray(row[6:1446], dtype=np.float32)
    tof = np.asarray(row[1446:1457], dtype=np.float32)
    if si.size != N_VOXEL or tof.size != 11:
        raise ValueError(f"bad feature size: si={si.size}, tof={tof.size}")
    return si, tof, label, beta


def load_balanced_sample(files: list[Path], events_per_class: int, seed: int):
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)

    si_by_label = {0: [], 1: []}
    tof_by_label = {0: [], 1: []}
    beta_by_label = {0: [], 1: []}
    bad = 0

    for fp in tqdm(shuffled, desc="load csv", dynamic_ncols=True):
        fallback = infer_particle_label(fp)
        if fallback in (0, 1) and len(si_by_label[fallback]) >= events_per_class:
            continue

        with fp.open() as f:
            for row in csv.reader(f):
                try:
                    si, tof, label, beta = parse_row(row)
                except Exception:
                    bad += 1
                    continue
                if label not in (0, 1):
                    if fallback not in (0, 1):
                        bad += 1
                        continue
                    label = fallback
                if len(si_by_label[label]) >= events_per_class:
                    continue
                si_by_label[label].append(si)
                tof_by_label[label].append(tof)
                beta_by_label[label].append(beta)

        if all(len(si_by_label[k]) >= events_per_class for k in (0, 1)):
            break

    for label in (0, 1):
        if len(si_by_label[label]) < events_per_class:
            raise RuntimeError(
                f"not enough label={label}: {len(si_by_label[label])} < {events_per_class}"
            )

    si = np.concatenate(
        [np.stack(si_by_label[0]), np.stack(si_by_label[1])], axis=0
    )
    tof = np.concatenate(
        [np.stack(tof_by_label[0]), np.stack(tof_by_label[1])], axis=0
    )
    beta = np.concatenate(
        [np.asarray(beta_by_label[0], dtype=np.float32), np.asarray(beta_by_label[1], dtype=np.float32)],
        axis=0,
    )
    y = np.asarray([0] * events_per_class + [1] * events_per_class, dtype=np.int64)

    order = np.random.default_rng(seed).permutation(y.size)
    return si[order], tof[order], beta[order], y[order], bad


def split_arrays(si, tof, beta, y, train_frac: float, val_frac: float):
    n = y.size
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }
    return {
        name: (si[sl], tof[sl], beta[sl], y[sl])
        for name, sl in splits.items()
    }


def make_features(si: np.ndarray, tof: np.ndarray, beta: np.ndarray, mode: str) -> np.ndarray:
    # Energy-like features are log-compressed; beta is only for diagnostics,
    # not included unless explicitly requested.
    si_log = np.log1p(np.clip(si, 0, None))
    tof_log = np.log1p(np.clip(tof, 0, None))

    if mode == "si":
        return si_log
    if mode == "tof":
        return tof_log
    if mode == "si_tof":
        return np.concatenate([si_log, tof_log], axis=1)
    if mode == "si_stats":
        q = np.quantile(si_log, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
        nz = (si > 0).sum(axis=1, keepdims=True).astype(np.float32)
        stats = np.stack(
            [
                si_log.sum(axis=1),
                si_log.mean(axis=1),
                si_log.std(axis=1),
                si_log.min(axis=1),
                si_log.max(axis=1),
            ],
            axis=1,
        )
        return np.concatenate([stats, q, nz], axis=1)
    if mode == "si_tof_beta":
        return np.concatenate([si_log, tof_log, beta.reshape(-1, 1)], axis=1)
    raise ValueError(f"unknown mode: {mode}")


def rejection_at_eff(y_true: np.ndarray, score: np.ndarray, eff: float) -> float:
    sig = score[y_true == 1]
    bkg = score[y_true == 0]
    if sig.size == 0 or bkg.size == 0:
        return float("nan")
    threshold = np.quantile(sig, 1.0 - eff)
    fpr = float((bkg >= threshold).mean())
    return float("inf") if fpr == 0 else 1.0 / fpr


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, score)),
        "rej50": rejection_at_eff(y_true, score, 0.50),
        "rej70": rejection_at_eff(y_true, score, 0.70),
        "rej90": rejection_at_eff(y_true, score, 0.90),
        "rej95": rejection_at_eff(y_true, score, 0.95),
        "rej98": rejection_at_eff(y_true, score, 0.98),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--glob", default="CNN*Atrest*.csv")
    ap.add_argument("--events-per-class", type=int, default=20000)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", default="si_stats,si,tof,si_tof")
    ap.add_argument("--models", default="hgb,logreg")
    ap.add_argument("--out-dir", default="results/nakagami_atrest_csv_baseline")
    args = ap.parse_args()

    np.random.seed(args.seed)
    files = list_files(args.inputs, args.glob)
    if not files:
        raise FileNotFoundError(args.inputs)
    print("files:", len(files))
    print("first:", files[0])

    si, tof, beta, y, bad = load_balanced_sample(files, args.events_per_class, args.seed)
    print("events:", len(y), "bad_rows:", bad, "labels:", np.bincount(y).tolist())
    print("beta:", float(beta.min()), float(beta.max()), float(beta.mean()))

    splits = split_arrays(si, tof, beta, y, args.train_frac, args.val_frac)
    for name, (_, _, _, yy) in splits.items():
        print(name, len(yy), np.bincount(yy).tolist())

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("mode,model,accuracy,auc,rej50,rej70,rej90,rej95,rej98")
    for mode in modes:
        x_train = make_features(*splits["train"][:3], mode)
        y_train = splits["train"][3]
        x_test = make_features(*splits["test"][:3], mode)
        y_test = splits["test"][3]

        for model_name in models:
            if model_name == "hgb":
                clf = HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    random_state=args.seed,
                )
            elif model_name == "logreg":
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, n_jobs=-1, random_state=args.seed),
                )
            else:
                raise ValueError(model_name)

            clf.fit(x_train, y_train)
            if hasattr(clf, "predict_proba"):
                score = clf.predict_proba(x_test)[:, 1]
            else:
                score = clf.decision_function(x_test)
            metrics = evaluate(y_test, score)
            row = {"mode": mode, "model": model_name, **metrics}
            rows.append(row)
            print(
                f"{mode},{model_name},"
                f"{metrics['accuracy']:.6f},{metrics['auc']:.6f},"
                f"{metrics['rej50']:.6g},{metrics['rej70']:.6g},"
                f"{metrics['rej90']:.6g},{metrics['rej95']:.6g},{metrics['rej98']:.6g}",
                flush=True,
            )

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(rows, f, indent=2)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()

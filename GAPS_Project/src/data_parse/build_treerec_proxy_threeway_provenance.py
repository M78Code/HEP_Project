#!/usr/bin/env python3
"""Derive three comparable populations from a common TreeRec candidate pool.

The input pool is selected with legacy at-rest plus truth Umbrella-to-Cube
topology.  It contains TreeMc strict-stop flags solely for this controlled
audit.  The proxy group is selected using *out-of-fold by ROOT source file*
scores from TreeRec hitseries observables, so a source file is never scored by
a proxy fitted on events from that same file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from GAPS_Project.src.data_parse.calibrate_treerec_hit_stop_proxy import (
    FEATURES,
    PARTICLES,
    logistic_matrix,
    read_features,
)


ARRAY_NAMES = (
    "labels.npy",
    "betas.npy",
    "random_seeds.npy",
    "chain_entries.npy",
    "source_file_indices.npy",
    "source_entries.npy",
)
GROUPS = (
    "summary_atrest_topology",
    "treerec_logistic_proxy",
    "truth_strict_topology",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def load_particle(source_dir: Path, particle: str) -> dict:
    directory = source_dir / particle
    required = (*ARRAY_NAMES, "source_files.txt", "truth_strict_stop.npy", "_SUCCESS")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"{directory}: missing {', '.join(missing)}")
    arrays = {name: np.load(directory / name, mmap_mode="r") for name in ARRAY_NAMES}
    lengths = {len(array) for array in arrays.values()}
    truth = np.load(directory / "truth_strict_stop.npy", mmap_mode="r")
    lengths.add(len(truth))
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise RuntimeError(f"{directory}: inconsistent or empty provenance arrays")
    files = [Path(line) for line in (directory / "source_files.txt").read_text().splitlines()
             if line.strip()]
    file_indices = np.asarray(arrays["source_file_indices.npy"], dtype=np.int64)
    if np.any(file_indices < 0) or np.any(file_indices >= len(files)):
        raise RuntimeError(f"{directory}: invalid source-file index")
    manifest = json.loads((directory / "export_manifest.json").read_text())
    provenance = {
        "file_indices": file_indices,
        "entries": np.asarray(arrays["source_entries.npy"], dtype=np.int64),
        "files": files,
    }
    return {
        "directory": directory,
        "arrays": arrays,
        "truth": np.asarray(truth, dtype=bool),
        "features": read_features(provenance),
        "manifest": manifest,
        "file_indices": file_indices,
    }


def oof_scores(pools: dict[str, dict], seed: int, requested_folds: int) -> tuple[np.ndarray, dict]:
    del seed  # GroupKFold is deterministic and source-file grouping is the control.
    feature_parts = {name: [] for name in FEATURES}
    truth_parts, particle_parts, source_parts = [], [], []
    for particle in PARTICLES:
        pool = pools[particle]
        for name in FEATURES:
            feature_parts[name].append(pool["features"][name])
        truth_parts.append(pool["truth"])
        particle_parts.append(np.full(len(pool["truth"]), particle, dtype=object))
        source_parts.append(np.asarray(
            [f"{particle}:{index}" for index in pool["file_indices"]], dtype=object))

    features = {name: np.concatenate(parts) for name, parts in feature_parts.items()}
    truth = np.concatenate(truth_parts)
    particles = np.concatenate(particle_parts)
    source_groups = np.concatenate(source_parts)
    n_groups = len(np.unique(source_groups))
    folds = min(requested_folds, n_groups)
    if folds < 2:
        raise RuntimeError("need at least two source ROOT files for an out-of-fold proxy")

    matrix = logistic_matrix(features)
    scores = np.full(len(truth), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(splitter.split(matrix, truth, source_groups), start=1):
        if len(np.unique(truth[train])) != 2:
            raise RuntimeError(f"fold {fold}: training data lacks one truth class")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260825),
        )
        model.fit(matrix[train], truth[train])
        scores[test] = model.predict_proba(matrix[test])[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("failed to produce all out-of-fold proxy scores")

    threshold = float(np.quantile(scores, 1.0 - float(truth.mean()), method="higher"))
    selected = scores >= threshold
    selected_count = int(selected.sum())
    strict_selected = int(np.count_nonzero(selected & truth))
    summary = {
        "source_file_grouped_folds": folds,
        "events": int(len(truth)),
        "truth_strict_fraction": float(truth.mean()),
        "out_of_fold_auc": float(roc_auc_score(truth, scores)),
        "threshold": threshold,
        "proxy_selected_fraction": float(selected.mean()),
        "precision_against_truth": float(strict_selected / selected_count),
        "recall_against_truth": float(strict_selected / int(truth.sum())),
        "per_particle": {},
    }
    offset = 0
    for particle in PARTICLES:
        count = len(pools[particle]["truth"])
        mask = selected[offset:offset + count]
        target = pools[particle]["truth"]
        summary["per_particle"][particle] = {
            "events": int(count),
            "truth_strict_fraction": float(target.mean()),
            "proxy_selected_fraction": float(mask.mean()),
            "proxy_truth_precision": float((mask & target).sum() / mask.sum()) if mask.any() else None,
            "proxy_truth_recall": float((mask & target).sum() / target.sum()) if target.any() else None,
        }
        pools[particle]["proxy_mask"] = mask
        offset += count
    return selected, summary


def write_population(
    destination: Path, pool: dict, mask: np.ndarray, selection: str, audit: dict,
) -> None:
    destination.mkdir(parents=True)
    for name, array in pool["arrays"].items():
        np.save(destination / name, np.asarray(array[mask]))
    shutil.copy2(pool["directory"] / "source_files.txt", destination / "source_files.txt")
    manifest = dict(pool["manifest"])
    manifest.update({
        "source": "derived from common legacy-atrest-toptrigger candidate pool",
        "selection": selection,
        "events": int(mask.sum()),
        "candidate_directory": str(pool["directory"]),
        "tree_mc_strict_stop_used_as_audit_label_only": True,
        "proxy_audit": audit,
    })
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    (destination / "_SUCCESS").write_text("ok\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.folds < 2:
        raise ValueError("--folds must be at least two")

    pools = {particle: load_particle(args.source_dir, particle) for particle in PARTICLES}
    _, proxy_audit = oof_scores(pools, args.seed, args.folds)
    args.output_dir.mkdir(parents=True)
    for particle in PARTICLES:
        pool = pools[particle]
        truth = pool["truth"]
        selections = {
            "summary_atrest_topology": np.ones(len(truth), dtype=bool),
            "treerec_logistic_proxy": pool["proxy_mask"],
            "truth_strict_topology": truth,
        }
        for group, mask in selections.items():
            if not mask.any():
                raise RuntimeError(f"{group}/{particle}: no selected events")
            write_population(
                args.output_dir / group / particle,
                pool,
                mask,
                group.replace("_", "-"),
                proxy_audit,
            )
            print(f"[{group}/{particle}] selected={int(mask.sum()):,}")

    (args.output_dir / "proxy_selection_manifest.json").write_text(
        json.dumps(proxy_audit, indent=2))
    (args.output_dir / "_SUCCESS").write_text("ok\n")
    print("\n===== TreeRec proxy three-way provenance =====")
    print(f"OOF AUC(truth strict): {proxy_audit['out_of_fold_auc']:.4f}")
    print(f"proxy threshold        : {proxy_audit['threshold']:.6g}")
    print(f"proxy precision/recall : {proxy_audit['precision_against_truth']:.4f} / "
          f"{proxy_audit['recall_against_truth']:.4f}")
    for particle, value in proxy_audit["per_particle"].items():
        print(f"{particle}: truth={value['truth_strict_fraction']:.2%}  "
              f"proxy={value['proxy_selected_fraction']:.2%}")
    print(f"complete: {args.output_dir}")


if __name__ == "__main__":
    main()

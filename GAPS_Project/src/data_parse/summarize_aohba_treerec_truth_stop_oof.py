#!/usr/bin/env python3
"""Validate complete OOF score coverage and report strict-stop proxy quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


PARTICLES = ("antiP", "antiD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_parts = []
    for fold in range(args.folds):
        path = args.score_dir / f"fold_{fold}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        score_parts.append(dict(np.load(path)))
    combined = {name: np.concatenate([part[name] for part in score_parts])
                for name in ("candidate_index", "particle", "truth_strict", "score")}
    result = {"folds": args.folds, "particles": {}}
    all_truth, all_score = [], []
    for particle_index, particle in enumerate(PARTICLES):
        expected_truth = np.asarray(np.load(
            args.provenance_dir / particle / "truth_strict_stop.npy", mmap_mode="r"), dtype=bool)
        mask = combined["particle"] == particle_index
        indices = combined["candidate_index"][mask].astype(np.int64)
        if len(indices) != len(expected_truth) or len(np.unique(indices)) != len(expected_truth):
            raise RuntimeError(f"{particle}: OOF scores do not cover every candidate exactly once")
        scores = np.empty(len(expected_truth), dtype=np.float32)
        scores[indices] = combined["score"][mask]
        observed = combined["truth_strict"][mask].astype(bool)
        if not np.array_equal(observed, expected_truth[indices]):
            raise RuntimeError(f"{particle}: saved truth labels disagree with candidates")
        all_truth.append(expected_truth)
        all_score.append(scores)
        result["particles"][particle] = {"scores": scores}
    truth = np.concatenate(all_truth)
    scores = np.concatenate(all_score)
    threshold = float(np.quantile(scores, 1.0 - float(truth.mean()), method="higher"))
    selected = scores >= threshold
    result["events"] = int(len(truth))
    result["truth_strict_fraction"] = float(truth.mean())
    result["out_of_fold_auc"] = float(roc_auc_score(truth, scores))
    result["threshold"] = threshold
    result["proxy_selected_fraction"] = float(selected.mean())
    result["precision"] = float((selected & truth).sum() / selected.sum())
    result["recall"] = float((selected & truth).sum() / truth.sum())
    per_particle = {}
    offset = 0
    for particle, target, score in zip(PARTICLES, all_truth, all_score):
        chosen = selected[offset:offset + len(target)]
        per_particle[particle] = {
            "truth_strict_fraction": float(target.mean()),
            "proxy_selected_fraction": float(chosen.mean()),
            "precision": float((chosen & target).sum() / chosen.sum()),
            "recall": float((chosen & target).sum() / target.sum()),
        }
        offset += len(target)
    result["per_particle"] = per_particle
    for item in result["particles"].values():
        del item["scores"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print("\n===== TreeRec full-graph strict-stop OOF summary =====")
    print(f"events              : {result['events']:,}")
    print(f"OOF AUC(truth)      : {result['out_of_fold_auc']:.6f}")
    print(f"selected fraction   : {result['proxy_selected_fraction']:.2%}")
    print(f"precision / recall  : {result['precision']:.4f} / {result['recall']:.4f}")
    for particle, value in per_particle.items():
        print(f"{particle}: selected={value['proxy_selected_fraction']:.2%}  "
              f"precision={value['precision']:.4f}  recall={value['recall']:.4f}")
    print(f"JSON: {args.output}")
    print("AOHBA TREEREC TRUTH-STOP OOF: COMPLETE")


if __name__ == "__main__":
    main()

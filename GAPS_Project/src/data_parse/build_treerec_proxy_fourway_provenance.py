#!/usr/bin/env python3
"""Build beta-matchable TreeRec stopping selections, including full-graph OOF.

The full-graph group is selected solely from source-file-held-out GravNet
scores.  TreeMc strict-stop is retained only as the supervised audit target
and as the controlled truth-selection comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from GAPS_Project.src.data_parse.build_treerec_proxy_threeway_provenance import (
    PARTICLES,
    load_particle,
    oof_scores as logistic_oof_scores,
    write_population,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--full-graph-score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--logistic-folds", type=int, default=5)
    parser.add_argument("--full-graph-folds", type=int, default=3)
    return parser.parse_args()


def load_full_graph_oof(
    pools: dict[str, dict], score_dir: Path, folds: int,
) -> dict:
    """Reassemble one source-held-out prediction for every candidate event."""
    parts = []
    for fold in range(folds):
        path = score_dir / f"fold_{fold}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append(dict(np.load(path)))
    required = ("candidate_index", "particle", "truth_strict", "score")
    if any(any(name not in part for name in required) for part in parts):
        raise RuntimeError("full-graph OOF score file has an invalid schema")
    combined = {
        name: np.concatenate([part[name] for part in parts]) for name in required
    }

    all_truth, all_scores = [], []
    per_particle = {}
    for particle_index, particle in enumerate(PARTICLES):
        truth = pools[particle]["truth"]
        mask = combined["particle"] == particle_index
        indices = combined["candidate_index"][mask].astype(np.int64)
        if len(indices) != len(truth) or len(np.unique(indices)) != len(truth):
            raise RuntimeError(
                f"{particle}: full-graph OOF scores do not cover candidates exactly once")
        observed_truth = combined["truth_strict"][mask].astype(bool)
        if not np.array_equal(observed_truth, truth[indices]):
            raise RuntimeError(f"{particle}: OOF score truth labels do not match provenance")
        scores = np.empty(len(truth), dtype=np.float32)
        scores[indices] = combined["score"][mask]
        pools[particle]["full_graph_score"] = scores
        all_truth.append(truth)
        all_scores.append(scores)

    truth = np.concatenate(all_truth)
    scores = np.concatenate(all_scores)
    threshold = float(np.quantile(scores, 1.0 - float(truth.mean()), method="higher"))
    selected = scores >= threshold
    offset = 0
    for particle in PARTICLES:
        target = pools[particle]["truth"]
        choice = selected[offset:offset + len(target)]
        pools[particle]["full_graph_mask"] = choice
        per_particle[particle] = {
            "truth_strict_fraction": float(target.mean()),
            "proxy_selected_fraction": float(choice.mean()),
            "precision_against_truth": float((choice & target).sum() / choice.sum()),
            "recall_against_truth": float((choice & target).sum() / target.sum()),
        }
        offset += len(target)
    strict_selected = int(np.count_nonzero(selected & truth))
    return {
        "score_origin": "three-fold source-file-held-out TreeRec GravNet",
        "events": int(len(truth)),
        "out_of_fold_auc": float(roc_auc_score(truth, scores)),
        "truth_strict_fraction": float(truth.mean()),
        "threshold": threshold,
        "proxy_selected_fraction": float(selected.mean()),
        "precision_against_truth": float(strict_selected / selected.sum()),
        "recall_against_truth": float(strict_selected / truth.sum()),
        "per_particle": per_particle,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    pools = {particle: load_particle(args.source_dir, particle) for particle in PARTICLES}
    _, logistic_audit = logistic_oof_scores(pools, args.seed, args.logistic_folds)
    full_graph_audit = load_full_graph_oof(
        pools, args.full_graph_score_dir, args.full_graph_folds)

    args.output_dir.mkdir(parents=True)
    for particle in PARTICLES:
        pool = pools[particle]
        truth = pool["truth"]
        selections = {
            "summary_atrest_topology": np.ones(len(truth), dtype=bool),
            "treerec_logistic_proxy": pool["proxy_mask"],
            "treerec_full_graph_oof_proxy": pool["full_graph_mask"],
            "truth_strict_topology": truth,
        }
        for group, mask in selections.items():
            if not mask.any():
                raise RuntimeError(f"{group}/{particle}: no selected events")
            audit = full_graph_audit if group == "treerec_full_graph_oof_proxy" else logistic_audit
            write_population(
                args.output_dir / group / particle,
                pool,
                mask,
                group.replace("_", "-"),
                audit,
            )
            print(f"[{group}/{particle}] selected={int(mask.sum()):,}")

    manifest = {
        "groups": [
            "summary_atrest_topology",
            "treerec_logistic_proxy",
            "treerec_full_graph_oof_proxy",
            "truth_strict_topology",
        ],
        "logistic_proxy_audit": logistic_audit,
        "full_graph_oof_proxy_audit": full_graph_audit,
        "truth_strict_stop_used_as_audit_label_only": True,
    }
    (args.output_dir / "proxy_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    (args.output_dir / "_SUCCESS").write_text("ok\n")
    print("\n===== TreeRec proxy four-way provenance =====")
    print(f"Logistic OOF AUC(truth): {logistic_audit['out_of_fold_auc']:.4f}")
    print(f"Full-graph OOF AUC(truth): {full_graph_audit['out_of_fold_auc']:.4f}")
    print("Full-graph proxy precision/recall: "
          f"{full_graph_audit['precision_against_truth']:.4f} / "
          f"{full_graph_audit['recall_against_truth']:.4f}")
    print(f"complete: {args.output_dir}")


if __name__ == "__main__":
    main()

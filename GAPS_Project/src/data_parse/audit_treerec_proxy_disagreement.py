#!/usr/bin/env python3
"""Audit where the TreeRec stopping proxy agrees and disagrees with truth.

The proxy is re-scored out of fold by ROOT source file, exactly as in the
three-way selection builder.  TreeMc flags remain audit-only labels; neither
the flags nor the proxy score are passed to GravNet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from GAPS_Project.src.data_parse.build_treerec_proxy_threeway_provenance import (
    FEATURES,
    PARTICLES,
    load_particle,
    oof_scores,
)


DISPLAY_FEATURES = (
    "n_hits",
    "n_tracker_hits",
    "tracker_energy",
    "tracker_energy_fraction",
    "tracker_mean_energy",
    "n_tracker_layers",
    "tracker_layer_span",
    "max_tracker_energy_fraction",
)
TRUTH_FLAGS = (
    "truth_summary_stopped.npy",
    "truth_kinetic_zero_in_tracker.npy",
    "truth_has_zero_step.npy",
    "truth_strict_stop.npy",
)
CATEGORIES = (
    ("true_positive", "strict + proxy", lambda truth, proxy: truth & proxy),
    ("false_negative", "strict + proxy missed", lambda truth, proxy: truth & ~proxy),
    ("false_positive", "not strict + proxy selected", lambda truth, proxy: ~truth & proxy),
    ("true_negative", "not strict + proxy rejected", lambda truth, proxy: ~truth & ~proxy),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def best_auc(values: np.ndarray, positive: np.ndarray) -> dict[str, float | str]:
    auc = float(roc_auc_score(positive, values))
    if auc < 0.5:
        return {"auc": 1.0 - auc, "direction": "lower for positive"}
    return {"auc": auc, "direction": "higher for positive"}


def load_audit_flags(pool: dict) -> dict[str, np.ndarray]:
    flags = {}
    for name in TRUTH_FLAGS:
        path = pool["directory"] / name
        if not path.is_file():
            raise RuntimeError(f"missing audit-only truth flag: {path}")
        values = np.asarray(np.load(path, mmap_mode="r"), dtype=bool)
        if len(values) != len(pool["truth"]):
            raise RuntimeError(f"{path}: length mismatch")
        flags[name.removesuffix(".npy")] = values
    return flags


def category_report(pool: dict, flags: dict[str, np.ndarray]) -> dict:
    truth = pool["truth"]
    proxy = pool["proxy_mask"]
    report = {}
    for name, label, make_mask in CATEGORIES:
        mask = make_mask(truth, proxy)
        feature_summary = {
            feature: summary(pool["features"][feature][mask])
            for feature in DISPLAY_FEATURES
        }
        report[name] = {
            "label": label,
            "events": int(mask.sum()),
            "fraction": float(mask.mean()),
            "mean_proxy_score": float(pool["proxy_score"][mask].mean()),
            "truth_flag_rates": {
                flag: float(values[mask].mean()) for flag, values in flags.items()
            },
            "feature_summary": feature_summary,
        }
    return report


def residual_feature_ranking(
    features: dict[str, np.ndarray], base_mask: np.ndarray, positive: np.ndarray
) -> list[dict]:
    result = []
    for feature in FEATURES:
        values = features[feature][base_mask]
        target = positive[base_mask]
        if len(np.unique(target)) != 2:
            continue
        metric = best_auc(values, target)
        result.append({"feature": feature, **metric})
    return sorted(result, key=lambda item: item["auc"], reverse=True)


def print_particle(particle: str, report: dict, captured: list[dict], purity: list[dict]) -> None:
    print(f"\n[{particle}]")
    print("category                       events    fraction  mean proxy score")
    for key, _, _ in CATEGORIES:
        value = report[key]
        print(f"{value['label']:30s} {value['events']:8,} {value['fraction']:10.2%} "
              f"{value['mean_proxy_score']:16.4f}")
    print("  key TreeRec medians: category                         mean Edep  total Edep  tracker hits")
    for key in ("true_positive", "false_negative", "false_positive"):
        value = report[key]
        features = value["feature_summary"]
        print(
            f"    {value['label']:42s} "
            f"{features['tracker_mean_energy']['median']:9.3f} "
            f"{features['tracker_energy']['median']:11.3f} "
            f"{features['n_tracker_hits']['median']:12.1f}"
        )
    false_positive_flags = report["false_positive"]["truth_flag_rates"]
    print(
        "  proxy-only truth-flag rates: "
        f"summary={false_positive_flags['truth_summary_stopped']:.2%}  "
        f"K=0={false_positive_flags['truth_kinetic_zero_in_tracker']:.2%}  "
        f"zero-step={false_positive_flags['truth_has_zero_step']:.2%}"
    )
    for title, ranking in (
        ("within truth-strict: captured vs missed", captured),
        ("within proxy-selected: strict vs not-strict", purity),
    ):
        print(f"  strongest remaining TreeRec features ({title}):")
        for item in ranking[:4]:
            print(f"    {item['feature']:29s} AUC={item['auc']:.4f}  {item['direction']}")


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    pools = {}
    for particle in PARTICLES:
        print(f"[LOAD] {particle}: TreeRec hitseries features and audit flags")
        pools[particle] = load_particle(args.provenance_dir, particle)
    _, proxy_summary = oof_scores(pools, args.seed, args.folds)

    results = {"proxy": proxy_summary, "particles": {}}
    print("\n===== TreeRec proxy disagreement audit =====")
    print("Truth strict-stop is audit-only. OOF scores are grouped by ROOT source file.")
    for particle in PARTICLES:
        pool = pools[particle]
        flags = load_audit_flags(pool)
        report = category_report(pool, flags)
        truth = pool["truth"]
        proxy = pool["proxy_mask"]
        captured = residual_feature_ranking(pool["features"], truth, proxy)
        purity = residual_feature_ranking(pool["features"], proxy, truth)
        results["particles"][particle] = {
            "categories": report,
            "residual_ranking": {
                "truth_strict_captured_vs_missed": captured,
                "proxy_selected_strict_vs_not_strict": purity,
            },
        }
        print_particle(particle, report, captured, purity)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nJSON: {args.output}")
    print("TREEREC PROXY DISAGREEMENT AUDIT: COMPLETE")


if __name__ == "__main__":
    main()

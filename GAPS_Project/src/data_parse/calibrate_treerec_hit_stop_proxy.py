#!/usr/bin/env python3
"""Calibrate hitseries-only stopping proxies against audit-only truth labels.

TreeMc is used only to supply ``truth_strict_stop.npy`` during export.  This
program reads TreeRec hitseries fields as candidate observables and reports
their ability to recover that label.  It never writes a model cache and does
not define the final reconstruction selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot


PARTICLES = ("antiP", "antiD")
HIT_BRANCHES = {
    "volume": "Rec/hitseries_/hitseries_.volume_id_",
    "energy": "Rec/hitseries_/hitseries_.energydep_",
}
FEATURES = ("n_hits", "n_tracker_hits", "tracker_energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--events-per-class", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def roc_auc(values: np.ndarray, target: np.ndarray) -> float | None:
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(values)
    return float((ranks[target].sum() - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def matched_rate_cut(values: np.ndarray, target: np.ndarray) -> dict:
    truth_rate = float(target.mean())
    threshold = float(np.quantile(values, 1.0 - truth_rate, method="higher"))
    chosen = values >= threshold
    positives = int(target.sum())
    selected = int(chosen.sum())
    true_selected = int(np.count_nonzero(chosen & target))
    return {
        "truth_strict_fraction": truth_rate,
        "threshold": threshold,
        "proxy_selected_fraction": float(chosen.mean()),
        "precision": float(true_selected / selected) if selected else None,
        "recall": float(true_selected / positives) if positives else None,
        "selected": selected,
        "true_selected": true_selected,
    }


def load_provenance(directory: Path, particle: str, limit: int) -> dict:
    source = directory / particle
    required = (
        "source_file_indices.npy",
        "source_entries.npy",
        "source_files.txt",
        "truth_strict_stop.npy",
        "_SUCCESS",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"{source}: missing {', '.join(missing)}")
    file_indices = np.load(source / "source_file_indices.npy", mmap_mode="r")
    entries = np.load(source / "source_entries.npy", mmap_mode="r")
    truth = np.load(source / "truth_strict_stop.npy", mmap_mode="r")
    count = min(limit, len(entries))
    if count == 0:
        raise RuntimeError(f"{source}: no events")
    if len(file_indices) != len(entries) or len(truth) != len(entries):
        raise RuntimeError(f"{source}: provenance array length mismatch")
    files = [Path(line) for line in (source / "source_files.txt").read_text().splitlines()
             if line.strip()]
    return {
        "file_indices": np.asarray(file_indices[:count], dtype=np.int64),
        "entries": np.asarray(entries[:count], dtype=np.int64),
        "truth": np.asarray(truth[:count], dtype=bool),
        "files": files,
    }


def read_features(provenance: dict) -> dict[str, np.ndarray]:
    file_indices = provenance["file_indices"]
    entries = provenance["entries"]
    rows: list[dict | None] = [None] * len(entries)
    for file_index in np.unique(file_indices):
        output_indices = np.flatnonzero(file_indices == file_index)
        root_entries = entries[output_indices]
        start, stop = int(root_entries.min()), int(root_entries.max()) + 1
        path = provenance["files"][int(file_index)]
        with uproot.open(path) as root_file:
            rec = root_file["TreeRec"]
            volume = rec[HIT_BRANCHES["volume"]].array(
                entry_start=start, entry_stop=stop, library="ak")
            energy = rec[HIT_BRANCHES["energy"]].array(
                entry_start=start, entry_stop=stop, library="ak")
        for output_index, entry in zip(output_indices, root_entries):
            local = int(entry) - start
            event_volume = np.asarray(volume[local], dtype=np.int64)
            event_energy = np.asarray(energy[local], dtype=np.float64)
            if len(event_volume) != len(event_energy):
                raise RuntimeError(f"{path}: hit-array mismatch at entry {entry}")
            tracker = (event_volume // 100_000_000) == 2
            rows[int(output_index)] = {
                "n_hits": float(len(event_volume)),
                "n_tracker_hits": float(tracker.sum()),
                "tracker_energy": float(event_energy[tracker].sum()),
            }
    if any(row is None for row in rows):
        raise RuntimeError("failed to read every TreeRec event")
    return {name: np.asarray([row[name] for row in rows], dtype=np.float64)
            for name in FEATURES}


def evaluate(features: dict[str, np.ndarray], truth: np.ndarray) -> dict:
    return {
        name: {
            "auc_truth_strict": roc_auc(values, truth),
            "matched_truth_rate_cut": matched_rate_cut(values, truth),
        }
        for name, values in features.items()
    }


def format_metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def print_results(results: dict) -> None:
    print("\n===== TreeRec hitseries proxy calibration =====")
    print("Truth strict-stop is an audit label only; it is not a proxy input.\n")
    for name, result in results.items():
        print(f"[{name}] N={result['events']:,} truth strict={result['truth_strict_fraction']:.2%}")
        print("feature                 AUC(truth)  threshold  selected  precision  recall")
        for feature, metric in result["features"].items():
            cut = metric["matched_truth_rate_cut"]
            print(
                f"{feature:23s} {format_metric(metric['auc_truth_strict']):>10s} "
                f"{cut['threshold']:10.4g} {cut['proxy_selected_fraction']:8.2%} "
                f"{format_metric(cut['precision']):>10s} {format_metric(cut['recall']):>7s}"
            )


def main() -> None:
    args = parse_args()
    if args.events_per_class < 1:
        raise ValueError("--events-per-class must be positive")
    results = {}
    combined_features = {name: [] for name in FEATURES}
    combined_truth = []
    for particle in PARTICLES:
        provenance = load_provenance(args.provenance_dir, particle, args.events_per_class)
        features = read_features(provenance)
        truth = provenance["truth"]
        result = evaluate(features, truth)
        results[particle] = {
            "events": int(len(truth)),
            "truth_strict_fraction": float(truth.mean()),
            "features": result,
        }
        for name in FEATURES:
            combined_features[name].append(features[name])
        combined_truth.append(truth)

    features = {name: np.concatenate(parts) for name, parts in combined_features.items()}
    truth = np.concatenate(combined_truth)
    results["combined_particles"] = {
        "events": int(len(truth)),
        "truth_strict_fraction": float(truth.mean()),
        "features": evaluate(features, truth),
    }
    print_results(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nJSON: {args.output}")
    print("TREEREC HITSERIES PROXY CALIBRATION: COMPLETE")


if __name__ == "__main__":
    main()

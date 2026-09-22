"""Evaluate traditional and hybrid reconstruction on chronological event blocks.

For each measured position, the lowest event IDs form train, the next block
forms validation, and the highest event IDs form the held-out test.  Model
seeds are repeated serially while the chronological test population is fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import Scintillator_Project
from Scintillator_Project.src.scripts.repeat_ohba_all_usable_splits import (
    ensure_cache,
    make_chronological_split,
    source_files,
    train_trial,
)


PROJECT_ROOT = Path(Scintillator_Project.__file__).resolve().parent


def parse_seeds(text: str) -> list[int]:
    try:
        seeds = [int(value) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("model seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("provide one or more unique model seeds")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "dataset/split")
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "dataset/cache/ohba_all_usable_events_roi450_800.npz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/ohba_all_usable_temporal_blocks",
    )
    parser.add_argument("--model-seeds", type=parse_seeds, default=[20260825, 20260826, 20260827])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def partition_audit(
    labels: np.ndarray,
    event_ids: np.ndarray,
    split: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for position in np.unique(labels):
        row: dict[str, object] = {"position_cm": float(position)}
        for name, indices in split.items():
            values = event_ids[indices[labels[indices] == position]]
            row[name] = {
                "events": int(values.size),
                "first_event_id": int(values.min()),
                "last_event_id": int(values.max()),
            }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_cache(source_files(args.split_dir), args.cache)
    with np.load(args.cache) as data:
        arrays = {name: data[name] for name in data.files}
    split = make_chronological_split(arrays["labels"], arrays["event_ids"])
    audit = partition_audit(arrays["labels"], arrays["event_ids"], split)
    (args.output_root / "chronological_partition_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    print("===== Chronological event-block benchmark =====", flush=True)
    print("Per position: earliest 70% train | next 15% validation | latest 15% test", flush=True)
    print(f"events={arrays['labels'].size} | model seeds={args.model_seeds}", flush=True)
    for row in audit:
        print(
            f"position={row['position_cm']:>5.0f} cm | "
            f"train {row['train']['first_event_id']}-{row['train']['last_event_id']} | "
            f"val {row['val']['first_event_id']}-{row['val']['last_event_id']} | "
            f"test {row['test']['first_event_id']}-{row['test']['last_event_id']}",
            flush=True,
        )

    results = []
    for model_seed in args.model_seeds:
        args.model_seed = model_seed
        trial_dir = args.output_root / f"model_{model_seed}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TRIAL START] model_seed={model_seed}", flush=True)
        report = train_trial(arrays, split, args, trial_dir)
        traditional_sigma = report["traditional"]["gaussian_fit"]["sigma_cm"]
        hybrid_sigma = report["hybrid"]["gaussian_fit"]["sigma_cm"]
        row = {
            "model_seed": model_seed,
            "traditional_sigma_cm": traditional_sigma,
            "hybrid_sigma_cm": hybrid_sigma,
            "improvement_cm": traditional_sigma - hybrid_sigma,
            "traditional_rmse_cm": report["traditional"]["rmse_cm"],
            "hybrid_rmse_cm": report["hybrid"]["rmse_cm"],
        }
        results.append(row)
        print(
            f"[TRIAL DONE] seed={model_seed} | traditional={traditional_sigma:.3f} cm | "
            f"hybrid={hybrid_sigma:.3f} cm | improvement={row['improvement_cm']:.3f} cm",
            flush=True,
        )

    improvements = np.asarray([row["improvement_cm"] for row in results])
    summary = {
        "protocol": "per-position chronological event-ID blocks; train-only traditional calibration",
        "cache": str(args.cache),
        "trials": results,
        "improvement_mean_cm": float(improvements.mean()),
        "improvement_sample_std_cm": float(improvements.std(ddof=1)) if improvements.size > 1 else 0.0,
        "all_trials_hybrid_better": bool(np.all(improvements > 0.0)),
    }
    output = args.output_root / "chronological_block_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("===== Chronological-block summary =====", flush=True)
    for row in results:
        print(
            f"seed={row['model_seed']} | traditional={row['traditional_sigma_cm']:.3f} | "
            f"hybrid={row['hybrid_sigma_cm']:.3f} | improvement={row['improvement_cm']:.3f}",
            flush=True,
        )
    print(
        f"mean improvement={summary['improvement_mean_cm']:.3f} +/- "
        f"{summary['improvement_sample_std_cm']:.3f} cm | "
        f"all hybrid better={summary['all_trials_hybrid_better']}",
        flush=True,
    )
    print(f"saved: {output}", flush=True)
    print("OHBA CHRONOLOGICAL EVENT-BLOCK BENCHMARK: COMPLETE", flush=True)


if __name__ == "__main__":
    main()

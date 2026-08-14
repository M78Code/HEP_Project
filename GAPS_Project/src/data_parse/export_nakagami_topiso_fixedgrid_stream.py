#!/usr/bin/env python3
"""Stream Nakagami topIso Atrest CSV files into fixed-grid GNN npy arrays.

This exporter is for large fixed-grid CSV samples such as the Nakagami
40M Atrest pool:

  CNN210729_Dbar_isot_200K_beta02to05_Atrest_000.csv ... _099.csv
  CNN210729_Pbar_isot_200K_beta02to05_Atrest_000.csv ... _099.csv

Unlike export_nakagami_atrest_voxel_gnn_input.py, this script does not collect
all event references in memory.  It assigns files to train/val/test by index
range and writes rows directly into memmapped .npy arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


N_VOXEL = 10 * 12 * 12
N_TOF_PADDLE = 172


@dataclass(frozen=True)
class Layout:
    n_cols: int
    label_col: int
    beta_col: int | None
    si_start: int
    si_end: int
    tof_primary_start: int
    tof_primary_end: int

    @property
    def tof_primary_dim(self) -> int:
        return self.tof_primary_end - self.tof_primary_start


LAYOUTS = {
    "topiso1457": Layout(
        n_cols=1457,
        label_col=2,
        beta_col=4,
        si_start=6,
        si_end=1446,
        tof_primary_start=1446,
        tof_primary_end=1457,
    ),
}


def parse_range(text: str) -> range:
    try:
        start_s, stop_s = text.split(":", 1)
        start = int(start_s)
        stop = int(stop_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"range must be START:STOP, got {text!r}"
        ) from exc
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError(
            f"range must satisfy 0 <= START < STOP, got {text!r}"
        )
    return range(start, stop)


def particle_files(csv_dir: Path, particle: str, layout_tag: str) -> list[Path]:
    if particle == "dbar":
        pattern = f"CNN*Dbar*{layout_tag}*.csv"
    elif particle == "pbar":
        pattern = f"CNN*Pbar*{layout_tag}*.csv"
    else:
        raise ValueError(particle)
    return sorted(csv_dir.glob(pattern))


def select_files(files: list[Path], idx_range: range, particle: str) -> list[Path]:
    stop = idx_range.stop
    if len(files) < stop:
        raise FileNotFoundError(
            f"not enough {particle} files: need index {stop - 1}, found {len(files)}"
        )
    return [files[i] for i in idx_range]


def count_lines(path: Path) -> int:
    n = 0
    with path.open() as handle:
        for _ in handle:
            n += 1
    return n


def allocate_arrays(split_dir: Path, n_events: int, layout: Layout):
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "voxels": np.lib.format.open_memmap(
            split_dir / "voxels.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_events, 10, 12, 12),
        ),
        "tof_paddles": np.lib.format.open_memmap(
            split_dir / "tof_paddles.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_events, N_TOF_PADDLE),
        ),
        "tof_primary": np.lib.format.open_memmap(
            split_dir / "tof_primary.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_events, layout.tof_primary_dim),
        ),
        "labels": np.lib.format.open_memmap(
            split_dir / "labels.npy",
            mode="w+",
            dtype=np.int64,
            shape=(n_events,),
        ),
        "betas": np.lib.format.open_memmap(
            split_dir / "betas.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_events,),
        ),
    }


def flush_arrays(arrays) -> None:
    for array in arrays.values():
        array.flush()


def row_to_arrays(row: list[str], layout: Layout):
    if len(row) != layout.n_cols:
        raise ValueError(f"bad column count: {len(row)} != {layout.n_cols}")

    label = int(float(row[layout.label_col]))
    if label not in (0, 1):
        raise ValueError(f"bad label: {label}")

    beta = 0.0 if layout.beta_col is None else float(row[layout.beta_col])

    si = np.asarray(row[layout.si_start : layout.si_end], dtype=np.float32)
    if si.size != N_VOXEL:
        raise ValueError(f"bad Si size: {si.size}")

    tof_primary = np.asarray(
        row[layout.tof_primary_start : layout.tof_primary_end],
        dtype=np.float32,
    )
    if tof_primary.size != layout.tof_primary_dim:
        raise ValueError(f"bad TOF primary size: {tof_primary.size}")

    return si.reshape(10, 12, 12), tof_primary, label, beta


def next_valid_row(reader, layout: Layout, expected_label: int, state: dict):
    for row in reader:
        state["seen"] += 1
        try:
            voxel, tof_primary, label, beta = row_to_arrays(row, layout)
        except Exception:
            state["bad"] += 1
            continue
        if label != expected_label:
            state["bad"] += 1
            continue
        return voxel, tof_primary, label, beta
    return None


def export_interleaved_file_pairs(
    *,
    dbar_files: list[Path],
    pbar_files: list[Path],
    arrays,
    layout: Layout,
    rows_per_file: int,
    split: str,
) -> tuple[int, dict, dict]:
    if len(dbar_files) != len(pbar_files):
        raise RuntimeError(
            f"{split}: Dbar/Pbar file count mismatch "
            f"{len(dbar_files)} != {len(pbar_files)}"
        )

    cursor = 0
    dbar_summary = {"written": 0, "bad_rows": 0, "discarded_rows": 0, "files": []}
    pbar_summary = {"written": 0, "bad_rows": 0, "discarded_rows": 0, "files": []}

    for dbar_path, pbar_path in zip(dbar_files, pbar_files):
        d_state = {"seen": 0, "bad": 0, "written": 0}
        p_state = {"seen": 0, "bad": 0, "written": 0}
        d_start = cursor

        with dbar_path.open() as d_handle, pbar_path.open() as p_handle:
            d_reader = csv.reader(d_handle)
            p_reader = csv.reader(p_handle)
            pbar = tqdm(
                range(rows_per_file),
                desc=f"{split}:{dbar_path.name}+{pbar_path.name}",
                leave=False,
                dynamic_ncols=True,
            )
            for _ in pbar:
                d_row = next_valid_row(d_reader, layout, 1, d_state)
                p_row = next_valid_row(p_reader, layout, 0, p_state)
                if d_row is None:
                    raise RuntimeError(
                        f"{dbar_path} ended after {d_state['written']:,} valid rows; "
                        f"expected {rows_per_file:,}"
                    )
                if p_row is None:
                    raise RuntimeError(
                        f"{pbar_path} ended after {p_state['written']:,} valid rows; "
                        f"expected {rows_per_file:,}"
                    )

                for row_data in (d_row, p_row):
                    voxel, tof_primary, label, beta = row_data
                    arrays["voxels"][cursor] = voxel
                    arrays["tof_paddles"][cursor] = 0.0
                    arrays["tof_primary"][cursor] = tof_primary
                    arrays["labels"][cursor] = label
                    arrays["betas"][cursor] = beta
                    cursor += 1

                d_state["written"] += 1
                p_state["written"] += 1

            for _ in d_reader:
                d_state["seen"] += 1
                dbar_summary["discarded_rows"] += 1
            for _ in p_reader:
                p_state["seen"] += 1
                pbar_summary["discarded_rows"] += 1

        dbar_summary["written"] += d_state["written"]
        dbar_summary["bad_rows"] += d_state["bad"]
        dbar_summary["files"].append(
            {
                "file": str(dbar_path),
                "start": d_start,
                "stop": d_start + rows_per_file * 2,
                "seen": d_state["seen"],
                "written": d_state["written"],
                "bad": d_state["bad"],
                "discarded_after_limit": max(0, d_state["seen"] - rows_per_file),
            }
        )

        pbar_summary["written"] += p_state["written"]
        pbar_summary["bad_rows"] += p_state["bad"]
        pbar_summary["files"].append(
            {
                "file": str(pbar_path),
                "start": d_start + 1,
                "stop": d_start + rows_per_file * 2,
                "seen": p_state["seen"],
                "written": p_state["written"],
                "bad": p_state["bad"],
                "discarded_after_limit": max(0, p_state["seen"] - rows_per_file),
            }
        )

    return cursor, dbar_summary, pbar_summary


def verify_split(split_dir: Path, expected_n: int, expected_each: int) -> dict:
    labels = np.load(split_dir / "labels.npy", mmap_mode="r")
    betas = np.load(split_dir / "betas.npy", mmap_mode="r")
    voxels = np.load(split_dir / "voxels.npy", mmap_mode="r")
    tof_primary = np.load(split_dir / "tof_primary.npy", mmap_mode="r")

    counts = {
        "0": int((np.asarray(labels) == 0).sum()),
        "1": int((np.asarray(labels) == 1).sum()),
    }
    out = {
        "n_events": int(len(labels)),
        "label_counts": counts,
        "beta_min": float(np.min(betas)) if len(betas) else None,
        "beta_max": float(np.max(betas)) if len(betas) else None,
        "voxels_shape": list(voxels.shape),
        "tof_primary_shape": list(tof_primary.shape),
    }
    if len(labels) != expected_n:
        raise RuntimeError(f"{split_dir}: n_events mismatch {len(labels)} != {expected_n}")
    if counts != {"0": expected_each, "1": expected_each}:
        raise RuntimeError(f"{split_dir}: label mismatch {counts}")
    return out


def export_split(
    *,
    split: str,
    dbar_files: list[Path],
    pbar_files: list[Path],
    output_dir: Path,
    output_suffix: str,
    rows_per_file: int,
    layout: Layout,
    overwrite: bool,
    verify: bool,
) -> dict:
    split_dir = output_dir / f"{split}_{output_suffix}"
    n_per_class = len(dbar_files) * rows_per_file
    n_events = n_per_class * 2

    if split_dir.exists() and not overwrite:
        raise FileExistsError(f"{split_dir} exists; pass --overwrite to replace it")

    arrays = allocate_arrays(split_dir, n_events, layout)
    started = time.time()

    try:
        cursor, dbar_summary, pbar_summary = export_interleaved_file_pairs(
            dbar_files=dbar_files,
            pbar_files=pbar_files,
            arrays=arrays,
            layout=layout,
            rows_per_file=rows_per_file,
            split=split,
        )
        if cursor != n_events:
            raise RuntimeError(f"{split}: wrote {cursor:,}, expected {n_events:,}")
    finally:
        flush_arrays(arrays)

    summary = {
        "split": split,
        "complete": True,
        "n_events": n_events,
        "events_per_class": n_per_class,
        "rows_per_file": rows_per_file,
        "label_counts": {"0": n_per_class, "1": n_per_class},
        "voxel_shape": [10, 12, 12],
        "tof_paddles": N_TOF_PADDLE,
        "tof_primary": layout.tof_primary_dim,
        "dbar": dbar_summary,
        "pbar": pbar_summary,
        "elapsed_sec": round(time.time() - started, 3),
    }
    if verify:
        summary["verify"] = verify_split(split_dir, n_events, n_per_class)

    with (split_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("saved", split_dir, json.dumps(summary["label_counts"]), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="topiso1457")
    parser.add_argument(
        "--layout-tag",
        default="Atrest",
        help="Filename tag used to select CSV files. Default: Atrest.",
    )
    parser.add_argument("--rows-per-file", type=int, default=200000)
    parser.add_argument("--train-range", type=parse_range, default=parse_range("0:80"))
    parser.add_argument("--val-range", type=parse_range, default=parse_range("80:90"))
    parser.add_argument("--test-range", type=parse_range, default=parse_range("90:100"))
    parser.add_argument("--output-suffix", default="nakagami_style_4M")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = LAYOUTS[args.layout]

    dbar_all = particle_files(args.csv_dir, "dbar", args.layout_tag)
    pbar_all = particle_files(args.csv_dir, "pbar", args.layout_tag)
    if not dbar_all or not pbar_all:
        raise FileNotFoundError(
            f"matched Dbar={len(dbar_all)}, Pbar={len(pbar_all)} under {args.csv_dir}"
        )

    selections = {
        "train": (
            select_files(dbar_all, args.train_range, "Dbar"),
            select_files(pbar_all, args.train_range, "Pbar"),
        ),
        "val": (
            select_files(dbar_all, args.val_range, "Dbar"),
            select_files(pbar_all, args.val_range, "Pbar"),
        ),
        "test": (
            select_files(dbar_all, args.test_range, "Dbar"),
            select_files(pbar_all, args.test_range, "Pbar"),
        ),
    }

    print("csv_dir:", args.csv_dir)
    print("layout:", args.layout)
    print("rows_per_file:", args.rows_per_file)
    print("matched files: Dbar", len(dbar_all), "Pbar", len(pbar_all))
    for split, (dbar_files, pbar_files) in selections.items():
        n = len(dbar_files) * args.rows_per_file * 2
        print(
            f"{split}: Dbar files={len(dbar_files)} Pbar files={len(pbar_files)} "
            f"events={n:,}"
        )

    if args.count_only:
        for split, (dbar_files, pbar_files) in selections.items():
            for particle, files in (("Dbar", dbar_files), ("Pbar", pbar_files)):
                too_short = []
                total_seen = 0
                for path in files:
                    n = count_lines(path)
                    total_seen += n
                    if n < args.rows_per_file:
                        too_short.append((str(path), n))
                print(
                    f"{split} {particle}: files={len(files)} rows_seen={total_seen:,} "
                    f"required={len(files) * args.rows_per_file:,} "
                    f"too_short={len(too_short)}"
                )
                if too_short:
                    print("  first too short:", too_short[0])
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for split, (dbar_files, pbar_files) in selections.items():
        all_summaries[split] = export_split(
            split=split,
            dbar_files=dbar_files,
            pbar_files=pbar_files,
            output_dir=args.output_dir,
            output_suffix=args.output_suffix,
            rows_per_file=args.rows_per_file,
            layout=layout,
            overwrite=args.overwrite,
            verify=args.verify,
        )

    summary = {
        "complete": True,
        "csv_dir": str(args.csv_dir),
        "layout": args.layout,
        "rows_per_file": args.rows_per_file,
        "output_suffix": args.output_suffix,
        "splits": all_summaries,
        "total_events": int(sum(x["n_events"] for x in all_summaries.values())),
    }
    with (args.output_dir / "export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("complete:", args.output_dir, "total_events", f"{summary['total_events']:,}")


if __name__ == "__main__":
    main()

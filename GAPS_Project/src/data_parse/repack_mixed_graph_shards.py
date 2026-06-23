"""Repack class-pure PyG graph shards into class-balanced mixed shards.

Each source antiD/antiP pair is processed in a fresh subprocess. This bounds
the lifetime of the large Python/PyG object graphs created by torch.load and
avoids retaining two class streams throughout training.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


SPLITS = ("train", "val", "test")
FORMAT_VERSION = 2


def load_graphs(path: Path) -> list:
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list) or not graphs:
        raise RuntimeError(
            f"invalid shard {path}: expected a non-empty list, "
            f"got {type(graphs).__name__}"
        )
    return graphs


def worker(args):
    anti_d = load_graphs(args.antiD)
    anti_p = load_graphs(args.antiP)
    if len(anti_d) != len(anti_p):
        raise RuntimeError(
            f"source shard sizes differ: antiD={len(anti_d):,}, "
            f"antiP={len(anti_p):,}"
        )

    if args.shuffle:
        random.Random(args.seed).shuffle(anti_d)
        random.Random(args.seed + 10_000).shuffle(anti_p)

    # Do not preserve storages inherited from the source archives. In
    # particular, tensors loaded through mmap-backed serialization can retain
    # non-resizable storages and fail when the repacked file is loaded later.
    mixed = []
    for graph_d, graph_p in zip(anti_d, anti_p):
        mixed.append(graph_d.clone())
        mixed.append(graph_p.clone())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(mixed, temporary)
    os.replace(temporary, args.output)

    summary = {
        "n_graphs": len(mixed),
        "label_counts": {"0": len(anti_p), "1": len(anti_d)},
        "source_antiD": str(args.antiD.resolve()),
        "source_antiP": str(args.antiP.resolve()),
        "seed": args.seed,
        "interleaved": True,
        "format_version": FORMAT_VERSION,
        "independent_tensor_storage": True,
    }
    with open(args.output.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(
        f"saved {args.output.name}: {len(mixed):,} graphs "
        f"({len(anti_d):,} per class)",
        flush=True,
    )
    # Large PyG object lists can crash while Python/C++ destructors run.
    # The worker has already atomically saved both output files, so terminate
    # the disposable subprocess without invoking interpreter cleanup.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def source_files(source_dir: Path, split: str, particle: str) -> list[Path]:
    files = sorted(source_dir.glob(f"{split}_{particle}_*.pt"))
    if not files:
        raise FileNotFoundError(
            f"no {split}_{particle}_*.pt under {source_dir}"
        )
    return files


def output_is_complete(path: Path) -> bool:
    summary_path = path.with_suffix(".json")
    if not path.exists() or not summary_path.exists():
        return False
    with open(summary_path, encoding="utf-8") as file:
        summary = json.load(file)
    return (
        int(summary.get("n_graphs", 0)) > 0
        and int(summary.get("format_version", 0)) == FORMAT_VERSION
        and summary.get("independent_tensor_storage") is True
    )


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paddle_ids = args.source_dir / "paddle_ids.json"
    if paddle_ids.exists():
        destination = args.output_dir / "paddle_ids.json"
        if not destination.exists():
            shutil.copy2(paddle_ids, destination)

    manifest = {
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "format": "alternating antiD/antiP graphs in each shard",
        "format_version": FORMAT_VERSION,
        "splits": {},
    }

    for split_index, split in enumerate(args.splits):
        anti_d_files = source_files(args.source_dir, split, "antiD")
        anti_p_files = source_files(args.source_dir, split, "antiP")
        if len(anti_d_files) != len(anti_p_files):
            raise RuntimeError(
                f"{split}: shard count differs: antiD={len(anti_d_files)}, "
                f"antiP={len(anti_p_files)}"
            )

        rows = []
        print(f"\n=== {split}: {len(anti_d_files)} shard pairs ===", flush=True)
        for index, (anti_d, anti_p) in enumerate(
            zip(anti_d_files, anti_p_files)
        ):
            output = args.output_dir / f"{split}_mixed_{index:03d}.pt"
            if output_is_complete(output):
                print(f"skip complete: {output.name}", flush=True)
            else:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--antiD",
                    str(anti_d),
                    "--antiP",
                    str(anti_p),
                    "--output",
                    str(output),
                    "--seed",
                    str(args.seed + split_index * 100_000 + index),
                ]
                if split == "train":
                    command.append("--shuffle")
                for attempt in range(1, args.max_retries + 1):
                    temporary = output.with_suffix(".pt.tmp")
                    temporary.unlink(missing_ok=True)
                    try:
                        subprocess.run(command, check=True)
                    except subprocess.CalledProcessError as error:
                        if output_is_complete(output):
                            print(
                                f"worker exited abnormally after completing "
                                f"{output.name}; keeping verified output",
                                flush=True,
                            )
                            break
                        if attempt == args.max_retries:
                            raise RuntimeError(
                                f"failed to create {output.name} after "
                                f"{args.max_retries} attempts"
                            ) from error
                        print(
                            f"worker failed for {output.name} "
                            f"(attempt {attempt}/{args.max_retries}); "
                            f"retrying in {args.retry_delay:.0f}s",
                            flush=True,
                        )
                        time.sleep(args.retry_delay)
                    else:
                        break

            with open(output.with_suffix(".json"), encoding="utf-8") as file:
                rows.append(json.load(file))

        manifest["splits"][split] = {
            "n_shards": len(rows),
            "n_graphs": sum(row["n_graphs"] for row in rows),
            "label_counts": {
                "0": sum(row["label_counts"]["0"] for row in rows),
                "1": sum(row["label_counts"]["1"] for row in rows),
            },
        }

    with open(
        args.output_dir / "mixed_manifest.json", "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

    print("\n========== complete ==========")
    for split, row in manifest["splits"].items():
        print(
            f"{split:5s}: {row['n_graphs']:,} graphs, "
            f"{row['n_shards']} mixed shards, labels={row['label_counts']}"
        )
    print(f"output: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--retry-delay", type=float, default=5.0)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--antiD", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--antiP", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--shuffle", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        for name in ("antiD", "antiP", "output"):
            if getattr(args, name) is None:
                parser.error(f"--{name} is required with --worker")
    elif args.source_dir is None or args.output_dir is None:
        parser.error("--source-dir and --output-dir are required")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        main(arguments)

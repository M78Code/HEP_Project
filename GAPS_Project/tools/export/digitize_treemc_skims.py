#!/usr/bin/env python3
"""Run resumable digitization-only jobs over exported TreeMc ROOT skims."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import uproot


REQUIRED_HITSERIES = (
    "Rec/hitseries_/hitseries_.volume_id_",
    "Rec/hitseries_/hitseries_.energydep_",
    "Rec/hitseries_/hitseries_.hit_position_",
    "Rec/hitseries_/hitseries_.hit_time_",
)


@dataclass(frozen=True)
class Job:
    input_path: Path
    output_path: Path
    log_path: Path
    status_path: Path
    seed_path: Path
    expected_events: int
    digitization_seed: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--crane", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--glob", default="*.root")
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--seed-base",
        type=int,
        help=(
            "derive a stable, distinct positive digitization seed for each "
            "input file; omit to preserve the Crane default"
        ),
    )
    return parser.parse_args()


def derive_digitization_seed(seed_base: int, input_name: str) -> int:
    """Derive a stable ROOT-compatible positive seed from a file name."""
    if seed_base <= 0:
        raise ValueError("--seed-base must be positive")
    payload = f"{seed_base}:{input_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "big") % 2_147_483_647
    return seed or 1


def tree_entries(path: Path, tree_name: str) -> int:
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"{path}: missing {tree_name}")
        return int(root_file[tree_name].num_entries)


def validate_output(job: Job) -> None:
    with uproot.open(job.output_path) as root_file:
        for tree_name in ("TreeMc", "TreeRec"):
            if tree_name not in root_file:
                raise RuntimeError(f"{job.output_path}: missing {tree_name}")
            entries = int(root_file[tree_name].num_entries)
            if entries != job.expected_events:
                raise RuntimeError(
                    f"{job.output_path}: {tree_name} has {entries}, "
                    f"expected {job.expected_events}"
                )
        rec = root_file["TreeRec"]
        for branch in REQUIRED_HITSERIES:
            if branch not in rec:
                raise RuntimeError(f"{job.output_path}: missing {branch}")


def completed(job: Job) -> bool:
    if not job.status_path.is_file() or not job.output_path.is_file():
        return False
    if job.status_path.read_text(encoding="utf-8").strip() != "0":
        return False
    if job.digitization_seed is not None:
        if not job.seed_path.is_file():
            return False
        if job.seed_path.read_text(encoding="utf-8").strip() != str(
            job.digitization_seed
        ):
            return False
    try:
        validate_output(job)
    except Exception:
        return False
    return True


def build_command(args: argparse.Namespace, job: Job) -> list[str]:
    command = [
        str(args.crane),
        "-i", str(job.input_path),
        "-o", str(job.output_path),
        "--nevents", str(job.expected_events),
        "--input-tree-name", "TreeMc",
        "--do-digitization", "1",
        "--do-reconstruction", "0",
        "--clone-mc", "1",
        "--keep-not-triggered", "1",
    ]
    if job.digitization_seed is not None:
        command.extend(
            ["--digitization-seed", str(job.digitization_seed)]
        )
    return command


def run_job(args: argparse.Namespace, job: Job) -> tuple[str, int, float]:
    if completed(job):
        print(f"[SKIP] {job.input_path.name}", flush=True)
        return job.input_path.name, 0, 0.0
    if job.output_path.exists():
        raise FileExistsError(
            f"incomplete output exists; inspect or remove it: {job.output_path}"
        )

    command = build_command(args, job)
    print(
        f"[START] {job.input_path.name} events={job.expected_events:,} "
        f"seed={job.digitization_seed}",
        flush=True,
    )
    started = time.monotonic()
    with job.log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.monotonic() - started
    status = result.returncode
    if status == 0:
        try:
            validate_output(job)
        except Exception as error:
            status = 97
            with job.log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nOUTPUT VALIDATION FAILED: {error}\n")
    job.status_path.write_text(f"{status}\n", encoding="utf-8")
    if status == 0 and job.digitization_seed is not None:
        job.seed_path.write_text(
            f"{job.digitization_seed}\n", encoding="utf-8"
        )
    print(
        f"[DONE] {job.input_path.name} status={status} "
        f"seconds={elapsed:.1f}",
        flush=True,
    )
    return job.input_path.name, status, elapsed


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.seed_base is not None and args.seed_base <= 0:
        raise ValueError("--seed-base must be positive")
    if not args.crane.is_file():
        raise FileNotFoundError(f"missing Crane executable: {args.crane}")
    paths = sorted(args.input_dir.glob(args.glob))
    if args.max_files is not None:
        paths = paths[:args.max_files]
    if not paths:
        raise FileNotFoundError(f"no inputs under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for path in paths:
        entries = tree_entries(path, "TreeMc")
        digitization_seed = (
            derive_digitization_seed(args.seed_base, path.name)
            if args.seed_base is not None
            else None
        )
        jobs.append(
            Job(
                input_path=path,
                output_path=args.output_dir / f"reco_{path.name}",
                log_path=args.log_dir / f"{path.stem}.log",
                status_path=args.log_dir / f"{path.stem}.status",
                seed_path=args.log_dir / f"{path.stem}.seed",
                expected_events=entries,
                digitization_seed=digitization_seed,
            )
        )
    seeds = [
        job.digitization_seed
        for job in jobs
        if job.digitization_seed is not None
    ]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("derived digitization seed collision")
    total_events = sum(job.expected_events for job in jobs)
    if args.expected_events is not None and total_events != args.expected_events:
        raise RuntimeError(
            f"input has {total_events:,} events, expected "
            f"{args.expected_events:,}"
        )
    print(
        f"digitization-only files={len(jobs)} events={total_events:,} "
        f"parallel={args.jobs}",
        flush=True,
    )

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_job, args, job): job for job in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())

    failed = [name for name, status, _ in results if status != 0]
    summary = {
        "processing_mode": "digitization_only",
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "files": len(jobs),
        "events": total_events,
        "parallel_jobs": args.jobs,
        "digitization_seed_base": args.seed_base,
        "digitization_seeds": {
            job.input_path.name: job.digitization_seed for job in jobs
        },
        "failed": failed,
    }
    (args.output_dir / "digitization_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} digitization jobs failed")
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

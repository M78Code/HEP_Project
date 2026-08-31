#!/usr/bin/env python3
"""Export TreeMc skims described by one or more selection manifests."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportJob:
    name: str
    entry_list: Path
    root_files: tuple[Path, ...]
    metadata_file: Path
    output_prefix: Path
    output_files: tuple[Path, ...]
    log_path: Path
    status_path: Path
    shards: int
    events: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--events-per-shard", type=int, default=167)
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def load_jobs(args: argparse.Namespace) -> list[ExportJob]:
    jobs = []
    names = set()
    for manifest_path in args.selection_manifest:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in payload["selections"]:
            name = (
                f"{row['split']}_{row['particle']}_{row['source_id']}_"
                f"selected{row['events']}"
            )
            if name in names:
                raise RuntimeError(f"duplicate manifest selection: {name}")
            names.add(name)
            events = int(row["events"])
            shards = math.ceil(events / args.events_per_shard)
            output_prefix = args.output_dir / name
            output_files = tuple(
                Path(f"{output_prefix}_shard{index:02d}.root")
                for index in range(shards)
            )
            jobs.append(
                ExportJob(
                    name=name,
                    entry_list=Path(row["entry_list"]),
                    root_files=tuple(Path(path) for path in row["root_files"]),
                    metadata_file=Path(row["metadata_file"]),
                    output_prefix=output_prefix,
                    output_files=output_files,
                    log_path=args.log_dir / f"{name}.log",
                    status_path=args.log_dir / f"{name}.status",
                    shards=shards,
                    events=events,
                )
            )
    return jobs


def validate_job(job: ExportJob) -> None:
    required = (job.entry_list, job.metadata_file, *job.root_files)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{job.name}: missing input {missing[0]}")
    entry_count = sum(
        1 for line in job.entry_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if entry_count != job.events:
        raise RuntimeError(
            f"{job.name}: manifest has {job.events} events but entry list "
            f"has {entry_count}"
        )


def completed(job: ExportJob) -> bool:
    if not job.status_path.is_file():
        return False
    if job.status_path.read_text(encoding="utf-8").strip() != "0":
        return False
    return all(path.is_file() and path.stat().st_size > 0 for path in job.output_files)


def run_job(exporter: Path, job: ExportJob) -> tuple[str, int, float]:
    if completed(job):
        print(f"[SKIP] {job.name}", flush=True)
        return job.name, 0, 0.0

    existing = [path for path in job.output_files if path.exists()]
    if existing:
        raise FileExistsError(
            f"{job.name}: incomplete output already exists: {existing[0]}"
        )

    command = [str(exporter)]
    for root_file in job.root_files:
        command.extend(("--input", str(root_file)))
    command.extend(
        (
            "--entry-list",
            str(job.entry_list),
            "--metadata-file",
            str(job.metadata_file),
            "--output-prefix",
            str(job.output_prefix),
            "--shards",
            str(job.shards),
        )
    )

    print(
        f"[START] {job.name} events={job.events} shards={job.shards} "
        f"input_files={len(job.root_files)}",
        flush=True,
    )
    start = time.monotonic()
    with job.log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - start
    status = result.returncode
    if status == 0 and not all(
        path.is_file() and path.stat().st_size > 0
        for path in job.output_files
    ):
        status = 97
        with job.log_path.open("a", encoding="utf-8") as log:
            log.write("\nERROR: exporter returned 0 but outputs are missing\n")
    job.status_path.write_text(f"{status}\n", encoding="utf-8")
    print(
        f"[DONE] {job.name} status={status} "
        f"wall_seconds={elapsed:.1f}",
        flush=True,
    )
    return job.name, status, elapsed


def main() -> None:
    args = parse_args()
    if args.events_per_shard < 1:
        raise ValueError("--events-per-shard must be positive")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if not args.exporter.is_file():
        raise FileNotFoundError(f"missing exporter: {args.exporter}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs(args)
    for job in jobs:
        validate_job(job)
    print(
        f"export jobs={len(jobs)} events={sum(job.events for job in jobs):,} "
        f"output_shards={sum(job.shards for job in jobs)} parallel={args.jobs}",
        flush=True,
    )

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_job, args.exporter, job): job
            for job in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())

    failed = [name for name, status, _ in results if status != 0]
    summary = {
        "selection_manifests": [
            str(path.resolve()) for path in args.selection_manifest
        ],
        "events_per_shard": args.events_per_shard,
        "parallel_jobs": args.jobs,
        "jobs": len(jobs),
        "events": sum(job.events for job in jobs),
        "output_shards": sum(job.shards for job in jobs),
        "failed": failed,
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if failed:
        raise RuntimeError(f"{len(failed)} exports failed: {failed}")
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

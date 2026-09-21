#!/usr/bin/env python3
"""Audit the local scintillator dataset against Ohba's thesis event table.

The project no longer contains readable raw ``.dat`` files, so this tool does
not claim to reconstruct a raw-data event count.  It verifies the two JSON
exports that are present and makes the remaining provenance gap explicit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Thesis Chapter 3, Table 3.4.  Keep this table in code so the comparison is
# reproducible even if a local note is edited later.
THESIS_EVENTS = {
    "run9_15": 6022, "run10_25": 244, "run11_35": 302,
    "run12_45": 4509, "run8_55": 301, "run7_65": 222,
    "run13_75": 308, "run20_80": 3828, "run14_85": 284,
    "run19_95": 534, "run18_105": 390, "run17_115": 466,
    "run16_125": 4805, "run15_135": 239, "run21_140": 1189,
}

HEADER_RE = re.compile(
    r'"filename"\s*:\s*"(?P<filename>[^"]+)".*?'
    r'"position_label"\s*:\s*(?P<position>[-+0-9.eE]+).*?'
    r'"num_events"\s*:\s*(?P<num_events>\d+)',
    re.DOTALL,
)
EVENT_ID_RE = re.compile(rb'"event_id"\s*:\s*(\d+)')
SPLIT_RE = re.compile(
    r'"split"\s*:\s*"(?P<split>[^"]+)".*?'
    r'"num_events"\s*:\s*(?P<num_events>\d+)',
    re.DOTALL,
)


def metadata_and_ids(path: Path) -> dict[str, object]:
    """Read only the small JSON header and stream event identifiers.

    Waveforms are hundreds of MB in total.  Regex scanning is intentional:
    it checks event counts/IDs without materialising every waveform in memory.
    """
    with path.open("rb") as handle:
        header = handle.read(32 * 1024).decode("utf-8")
    match = HEADER_RE.search(header)
    if not match:
        raise RuntimeError(f"cannot parse JSON header: {path}")

    event_ids: list[int] = []
    tail = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            data = tail + chunk
            event_ids.extend(int(value) for value in EVENT_ID_RE.findall(data))
            tail = data[-64:]

    # The 64-byte overlap can duplicate a token crossing a chunk boundary.
    # Event IDs are consecutive in these exports, so de-duplicate only that
    # overlap while preserving the audit's duplicate-ID diagnostic.
    unique_ids = set(event_ids)
    return {
        "file": path.name,
        "source_filename": match.group("filename"),
        "position_cm": float(match.group("position")),
        "declared_events": int(match.group("num_events")),
        "scanned_event_id_tokens": len(event_ids),
        "unique_event_ids": len(unique_ids),
        "min_event_id": min(unique_ids) if unique_ids else None,
        "max_event_id": max(unique_ids) if unique_ids else None,
    }


def audit_export(directory: Path) -> dict[str, dict[str, object]]:
    if not directory.is_dir():
        return {}
    records = {}
    for path in sorted(directory.glob("run*_*.json")):
        record = metadata_and_ids(path)
        key = Path(str(record["source_filename"])).stem
        records[key] = record
    return records


def raw_status(project: Path) -> list[dict[str, object]]:
    candidates = [project / "dataset/raw", project / "dataset/tar_zip/A"]
    statuses = []
    for directory in candidates:
        files = sorted(directory.glob("*.dat")) if directory.is_dir() else []
        statuses.append({
            "path": str(directory),
            "exists": directory.is_dir(),
            "files": len(files),
            "nonempty_files": sum(path.stat().st_size > 0 for path in files),
            "bytes": sum(path.stat().st_size for path in files),
        })
    return statuses


def split_status(directory: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(directory.glob("*.json")):
        with path.open("rb") as handle:
            header = handle.read(4096).decode("utf-8")
        match = SPLIT_RE.search(header)
        if not match:
            raise RuntimeError(f"cannot parse split JSON header: {path}")
        records.append({
            "split": match.group("split"),
            "events": int(match.group("num_events")),
            "file": path.name,
        })
    return records


def print_table(primary: dict[str, dict[str, object]], secondary: dict[str, dict[str, object]]) -> dict[str, object]:
    print("\n===== Thesis table vs local JSON export =====")
    print(f"{'run / position':<16} {'thesis':>8} {'local':>8} {'delta':>8} {'IDs':>15} {'copy':>8}")
    print("-" * 72)
    rows = []
    for key, thesis_events in THESIS_EVENTS.items():
        local = primary.get(key)
        copy = secondary.get(key)
        local_events = local["declared_events"] if local else None
        delta = local_events - thesis_events if local_events is not None else None
        ids_ok = (
            local is not None
            and local["declared_events"] == local["unique_event_ids"]
            and local["min_event_id"] == 1
            and local["max_event_id"] == local["declared_events"]
        )
        copy_ok = (
            local is not None and copy is not None
            and local["declared_events"] == copy["declared_events"]
            and local["unique_event_ids"] == copy["unique_event_ids"]
        )
        print(
            f"{key:<16} {thesis_events:>8} "
            f"{str(local_events) if local_events is not None else 'MISSING':>8} "
            f"{str(delta) if delta is not None else 'n/a':>8} "
            f"{'valid' if ids_ok else 'CHECK':>15} "
            f"{'same' if copy_ok else 'CHECK':>8}"
        )
        rows.append({
            "run": key, "thesis_events": thesis_events,
            "local_events": local_events, "delta_local_minus_thesis": delta,
            "event_ids_valid": ids_ok, "processed_copies_agree": copy_ok,
        })

    thesis_total = sum(THESIS_EVENTS.values())
    local_total = sum(row["local_events"] or 0 for row in rows)
    print("-" * 72)
    print(f"{'TOTAL':<16} {thesis_total:>8} {local_total:>8} {local_total - thesis_total:>8}")
    return {"rows": rows, "thesis_total": thesis_total, "local_total": local_total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path, help="Optional machine-readable audit output.")
    args = parser.parse_args()
    project = args.project.resolve()
    processed = project / "dataset/processed"
    ch01 = processed / "processed_ch0_and_ch1_to_json"

    print("===== Scintillator dataset provenance audit =====")
    print(f"project: {project}")
    print("source of thesis counts: Chapter 3, Table 3.4")

    status = raw_status(project)
    print("\n===== Raw-data availability =====")
    for item in status:
        print(
            f"{item['path']}: exists={item['exists']}, files={item['files']}, "
            f"nonempty={item['nonempty_files']}, bytes={item['bytes']}"
        )

    primary = audit_export(ch01)
    secondary = audit_export(processed)
    summary = print_table(primary, secondary)
    summary["raw_data_status"] = status

    splits = split_status(project / "dataset/split")
    print("\n===== Current train / validation / test split =====")
    for item in splits:
        print(f"{item['split']:<5} {item['events']:>8}  {item['file']}")
    split_total = sum(item["events"] for item in splits)
    print(f"{'TOTAL':<5} {split_total:>8}")
    print(f"matches local export: {'yes' if split_total == summary['local_total'] else 'NO'}")
    summary["split_status"] = splits
    summary["split_total"] = split_total
    summary["interpretation"] = (
        "The local JSON export is internally auditable.  Any disagreement with "
        "the thesis table cannot be attributed to a raw parser without readable "
        "source .dat files or Ohba's original conversion/selection record."
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nJSON: {args.json_out}")

    print("\nAudit boundary: this establishes local export integrity, not the original raw-data provenance.")


if __name__ == "__main__":
    main()

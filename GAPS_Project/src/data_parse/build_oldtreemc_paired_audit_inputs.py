#!/usr/bin/env python3
"""Build paired old-TreeMc and reconstructed-TreeRec audit inputs.

The selected ROOT skims contain exact entries from the old Nakagami validation
CSV sample.  Crane reconstruction preserves their order.  This script joins
the three representations using ``SelectionMetadata.source_entry`` and writes:

* a tiny Nakagami fixed-grid test split for the old sparse-voxel checkpoint;
* a sharded TreeRec graph cache for the current global-log checkpoint;
* JSONL provenance with one shared ``pair_index`` per event.

The TreeRec node normalizer is copied from the full training cache.  It is never
refitted on this audit sample.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from GAPS_Project.src.data_parse.graph_builder import GraphBuilder


N_VOXEL = 10 * 12 * 12
N_TOF_PADDLE = 172
N_TOF_PRIMARY = 11
PDG = {"antip": -2212, "antid": -1000010020}
LABEL = {"antip": 0, "antid": 1}
SOURCE_ID = {"antip": 1627528714, "antid": 1627550286}


@dataclass(frozen=True)
class PairRecord:
    pair_index: int
    particle: str
    source_id: int
    source_entry: int
    shard: int
    shard_entry: int
    reco_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--skim-dir", type=Path, required=True)
    parser.add_argument("--reco-dir", type=Path, required=True)
    parser.add_argument("--node-normalizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--expected-per-class", type=int, default=1000)
    return parser.parse_args()


def scalar_array(tree, branch: str) -> np.ndarray:
    return np.asarray(tree[branch].array(library="np")).reshape(-1)


def discover_records(args: argparse.Namespace) -> list[PairRecord]:
    records: list[PairRecord] = []
    pair_index = 0

    for particle in ("antip", "antid"):
        particle_count = 0
        skim_paths = sorted(
            args.skim_dir.glob(f"{particle}_selected*_shard*.root")
        )
        if not skim_paths:
            raise FileNotFoundError(
                f"no selected skims for {particle} under {args.skim_dir}"
            )

        for skim_path in skim_paths:
            shard_text = skim_path.stem.rsplit("shard", 1)[-1]
            shard = int(shard_text)
            reco_path = args.reco_dir / f"reco_{skim_path.name}"
            if not reco_path.exists():
                raise FileNotFoundError(f"missing reconstructed shard: {reco_path}")

            with uproot.open(skim_path) as skim_file:
                metadata = skim_file["SelectionMetadata"]
                source_entries = scalar_array(metadata, "source_entry")
                selection_indices = scalar_array(metadata, "selection_index")
                skim_entries = skim_file["TreeMc"].num_entries

            with uproot.open(reco_path) as reco_file:
                mc_entries = reco_file["TreeMc"].num_entries
                rec_entries = reco_file["TreeRec"].num_entries

            if not (skim_entries == mc_entries == rec_entries == len(source_entries)):
                raise RuntimeError(
                    f"entry mismatch for {particle} shard {shard:02d}: "
                    f"skim={skim_entries}, mc={mc_entries}, rec={rec_entries}, "
                    f"metadata={len(source_entries)}"
                )
            if len(np.unique(selection_indices)) != len(selection_indices):
                raise RuntimeError(f"duplicate selection_index in {skim_path}")

            for local_entry, source_entry in enumerate(source_entries):
                records.append(
                    PairRecord(
                        pair_index=pair_index,
                        particle=particle,
                        source_id=SOURCE_ID[particle],
                        source_entry=int(source_entry),
                        shard=shard,
                        shard_entry=local_entry,
                        reco_path=reco_path,
                    )
                )
                pair_index += 1
                particle_count += 1

        if particle_count != args.expected_per_class:
            raise RuntimeError(
                f"expected {args.expected_per_class} {particle} events, "
                f"found {particle_count}"
            )

    keys = [(record.source_id, record.source_entry) for record in records]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate (source_id, source_entry) pair")
    return records


def load_csv_rows(csv_dir: Path, records: list[PairRecord]):
    targets = {(record.source_id, record.source_entry) for record in records}
    found: dict[tuple[int, int], dict[str, object]] = {}

    paths = sorted(csv_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no CSV files under {csv_dir}")

    for path in paths:
        with path.open() as handle:
            for row_number, line in enumerate(handle):
                prefix = line.split(",", 2)
                if len(prefix) < 3:
                    continue
                try:
                    key = (int(float(prefix[0])), int(float(prefix[1])))
                except ValueError:
                    continue
                if key not in targets:
                    continue

                row = line.rstrip("\n").split(",")
                if len(row) != 1457:
                    raise RuntimeError(
                        f"target row {key} has {len(row)} columns in {path}"
                    )
                if key in found:
                    raise RuntimeError(f"duplicate target row {key} in {path}")

                voxel = np.asarray(row[6:1446], dtype=np.float32)
                tof_primary = np.asarray(row[1446:1457], dtype=np.float32)
                if voxel.size != N_VOXEL or tof_primary.size != N_TOF_PRIMARY:
                    raise RuntimeError(f"bad old fixed-grid row {key} in {path}")
                found[key] = {
                    "label": int(float(row[2])),
                    "beta": float(row[4]),
                    "voxel": voxel.reshape(10, 12, 12),
                    "tof_primary": tof_primary,
                    "csv_path": str(path),
                    "csv_row": row_number,
                }

        print(
            f"scanned {path.name}: matched {len(found):,}/{len(targets):,}",
            flush=True,
        )
        if len(found) == len(targets):
            break

    missing = sorted(targets - set(found))
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} selected events in validation CSV; "
            f"first={missing[:5]}"
        )
    return found


def load_normalizer(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "global_log":
        raise ValueError(f"{path} is not a global_log node normalizer")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError(f"bad normalizer dimensions in {path}")
    return mean, std, payload


def positions_for_event(value) -> np.ndarray:
    if len(value) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(
        [
            np.asarray(value["fX"], dtype=np.float32),
            np.asarray(value["fY"], dtype=np.float32),
            np.asarray(value["fZ"], dtype=np.float32),
        ],
        axis=1,
    )


def build_outputs(
    args: argparse.Namespace,
    records: list[PairRecord],
    csv_rows: dict[tuple[int, int], dict[str, object]],
) -> None:
    fixed_dir = args.output_dir / "old_treemc_fixedgrid"
    fixed_test_dir = fixed_dir / "test_nakagami_style_4M"
    treerec_dir = args.output_dir / "treerec_global_log"
    fixed_test_dir.mkdir(parents=True)
    treerec_dir.mkdir(parents=True)

    n_events = len(records)
    voxels = np.lib.format.open_memmap(
        fixed_test_dir / "voxels.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_events, 10, 12, 12),
    )
    tof_paddles = np.lib.format.open_memmap(
        fixed_test_dir / "tof_paddles.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_events, N_TOF_PADDLE),
    )
    tof_primary = np.lib.format.open_memmap(
        fixed_test_dir / "tof_primary.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_events, N_TOF_PRIMARY),
    )
    labels = np.lib.format.open_memmap(
        fixed_test_dir / "labels.npy",
        mode="w+",
        dtype=np.int64,
        shape=(n_events,),
    )
    betas = np.lib.format.open_memmap(
        fixed_test_dir / "betas.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_events,),
    )
    tof_paddles[:] = 0.0

    mean, std, normalizer_payload = load_normalizer(args.node_normalizer)
    builder = GraphBuilder(
        k=args.k,
        normalize=True,
        normalization_mode="global_log",
        global_feature_mean=mean,
        global_feature_std=std,
    )

    records_by_root: dict[Path, list[PairRecord]] = {}
    for record in records:
        records_by_root.setdefault(record.reco_path, []).append(record)

    graphs_by_pair: dict[int, object] = {}
    provenance: list[dict[str, object] | None] = [None] * n_events
    max_beta_delta = 0.0

    for reco_path, root_records in records_by_root.items():
        with uproot.open(reco_path) as root_file:
            mc = root_file["TreeMc"]
            rec = root_file["TreeRec"]
            pdgs = scalar_array(mc, "Mc/primaryPdg_")
            mc_betas = scalar_array(mc, "Mc/CEventBase/primaryBetaGenerated_")
            volume = rec["Rec/hitseries_/hitseries_.volume_id_"].array(library="ak")
            energy = rec["Rec/hitseries_/hitseries_.energydep_"].array(library="ak")
            position = rec["Rec/hitseries_/hitseries_.hit_position_"].array(library="ak")
            times = rec["Rec/hitseries_/hitseries_.hit_time_"].array(library="ak")

            for record in root_records:
                index = record.pair_index
                local = record.shard_entry
                key = (record.source_id, record.source_entry)
                csv_row = csv_rows[key]
                expected_label = LABEL[record.particle]
                expected_pdg = PDG[record.particle]

                if int(csv_row["label"]) != expected_label:
                    raise RuntimeError(f"CSV label mismatch for {key}")
                if int(pdgs[local]) != expected_pdg:
                    raise RuntimeError(f"TreeMc PDG mismatch for {key}")
                beta_delta = abs(float(csv_row["beta"]) - float(mc_betas[local]))
                max_beta_delta = max(max_beta_delta, beta_delta)
                if beta_delta > 5e-6:
                    raise RuntimeError(
                        f"CSV/TreeMc beta mismatch for {key}: delta={beta_delta}"
                    )

                event_volume = np.asarray(volume[local], dtype=np.int64)
                event_energy = np.asarray(energy[local], dtype=np.float32)
                event_position = positions_for_event(position[local])
                event_times = np.asarray(times[local], dtype=np.float32)
                if not (
                    len(event_volume)
                    == len(event_energy)
                    == len(event_position)
                    == len(event_times)
                ):
                    raise RuntimeError(f"TreeRec hit-array mismatch for {key}")
                if len(event_energy) <= 1:
                    raise RuntimeError(f"TreeRec event has <=1 hit for {key}")

                graph = builder.build_from_dict(
                    {
                        "energy": event_energy,
                        "positions": event_position,
                        "times": event_times,
                        "volume_id": event_volume,
                        "label": expected_pdg,
                        "beta": float(mc_betas[local]),
                    }
                )
                graph.pair_index = torch.tensor([index], dtype=torch.long)
                graph.source_id = torch.tensor([record.source_id], dtype=torch.long)
                graph.source_entry = torch.tensor(
                    [record.source_entry], dtype=torch.long
                )
                graphs_by_pair[index] = graph

                voxels[index] = csv_row["voxel"]
                tof_primary[index] = csv_row["tof_primary"]
                labels[index] = expected_label
                betas[index] = float(csv_row["beta"])
                provenance[index] = {
                    "pair_index": index,
                    "particle": record.particle,
                    "label": expected_label,
                    "source_id": record.source_id,
                    "source_entry": record.source_entry,
                    "shard": record.shard,
                    "shard_entry": local,
                    "reco_path": str(reco_path),
                    "csv_path": csv_row["csv_path"],
                    "csv_row": csv_row["csv_row"],
                    "mc_beta": float(mc_betas[local]),
                    "csv_beta": float(csv_row["beta"]),
                    "treerec_hits": len(event_energy),
                }

        print(f"built {reco_path.name}: {len(root_records):,} pairs", flush=True)

    graphs = [graphs_by_pair[index] for index in range(n_events)]
    graph_path = treerec_dir / "test_oldtreemc_paired_000.pt"
    torch.save(graphs, graph_path)
    (graph_path.with_suffix(".json")).write_text(
        json.dumps({"n_graphs": n_events, "paired": True}, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(args.node_normalizer, treerec_dir / "node_feature_normalizer.json")

    for array in (voxels, tof_paddles, tof_primary, labels, betas):
        array.flush()

    provenance_path = args.output_dir / "pair_provenance.jsonl"
    with provenance_path.open("w", encoding="utf-8") as handle:
        for item in provenance:
            if item is None:
                raise RuntimeError("internal error: missing provenance row")
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    manifest = {
        "events": n_events,
        "label_counts": {
            str(label): int(np.count_nonzero(np.asarray(labels) == label))
            for label in (0, 1)
        },
        "source_ids": SOURCE_ID,
        "old_fixedgrid_dir": str(fixed_dir),
        "treerec_cache_dir": str(treerec_dir),
        "node_normalizer": str(args.node_normalizer),
        "node_normalizer_payload": normalizer_payload,
        "max_abs_csv_treemc_beta_delta": max_beta_delta,
        "pair_provenance": str(provenance_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"complete: {args.output_dir}")
    print(f"pairs: {n_events:,} | labels: {manifest['label_counts']}")
    print(f"max |CSV beta - TreeMc beta|: {max_beta_delta:.3g}")


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    records = discover_records(args)
    print(f"selected records: {len(records):,}", flush=True)
    csv_rows = load_csv_rows(args.csv_dir, records)
    build_outputs(args, records, csv_rows)


if __name__ == "__main__":
    main()

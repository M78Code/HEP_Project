#!/usr/bin/env python3
"""Score every candidate in one held-out ROOT-file fold with a strict-stop GNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from GAPS_Project.src.data_parse.build_aohba_treerec_truth_stop_oof_fold_cache import (
    PARTICLES,
    source_fold_masks,
)
from GAPS_Project.src.data_parse.build_aohba_treerec_truth_stop_selector_cache import (
    iter_events,
    load_pool,
    make_provenance,
)
from GAPS_Project.src.models.gravnet import GravNetClassifier
from GAPS_Project.src.models.tree_rec_features import build_base_graph_feat
from GAPS_Project.src.data_parse.graph_builder import GraphBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> GravNetClassifier:
    model = GravNetClassifier(
        in_channels=8, hidden_dim=128, graph_feat_dim=45,
        num_blocks=6, normalization="batch")
    state = torch.load(path, map_location=device, weights_only=True)
    state = {key.replace("_orig_mod.", "").replace("module.", ""): value
             for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.folds < 3 or not 0 <= args.fold_index < args.folds:
        raise ValueError("invalid fold configuration")
    normalizer = json.loads(args.normalizer.read_text())
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    builder = GraphBuilder(
        k=8, normalize=True, normalization_mode="global_log",
        global_feature_mean=mean, global_feature_std=std)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    output_indices, output_particles, output_truth, output_scores = [], [], [], []
    for particle_index, particle in enumerate(PARTICLES):
        pool = load_pool(args.provenance_dir, particle)
        masks, files = source_fold_masks(
            pool["file_indices"], args.folds, args.fold_index,
            args.seed + (0 if particle == "antiP" else 1))
        candidate_indices = np.flatnonzero(masks["test"])
        selected = np.column_stack((candidate_indices, pool["truth"][candidate_indices].astype(np.int64)))
        provenance = make_provenance(pool, selected)
        print(f"[SCORE] fold={args.fold_index} {particle}: {len(candidate_indices):,} candidates "
              f"from source files {files['test']}")
        offset = 0
        for events in iter_events(provenance, PARTICLES[particle]["raw_name"], args.chunk_size):
            graphs = []
            for event in events:
                graph = builder.build_from_dict(event)
                graph.y = torch.tensor([event["truth_stop_target"]], dtype=torch.long)
                graphs.append(graph)
            loader = DataLoader(graphs, batch_size=args.batch_size, num_workers=0, pin_memory=True)
            chunk_scores = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logits = model(batch.x, batch.edge_index, batch.batch,
                                   graph_feat=build_base_graph_feat(batch))
                    chunk_scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            scores = np.concatenate(chunk_scores)
            size = len(scores)
            output_indices.append(candidate_indices[offset:offset + size])
            output_particles.append(np.full(size, particle_index, dtype=np.int8))
            output_truth.append(pool["truth"][candidate_indices[offset:offset + size]].astype(np.uint8))
            output_scores.append(scores.astype(np.float32))
            offset += size
        if offset != len(candidate_indices):
            raise RuntimeError(f"{particle}: score coverage mismatch")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        candidate_index=np.concatenate(output_indices),
        particle=np.concatenate(output_particles),
        truth_strict=np.concatenate(output_truth),
        score=np.concatenate(output_scores),
        fold=np.full(sum(len(item) for item in output_indices), args.fold_index, dtype=np.int16),
    )
    print(f"saved: {args.output}")
    print("AOHBA TREEREC TRUTH-STOP OOF FOLD SCORE: COMPLETE")


if __name__ == "__main__":
    main()

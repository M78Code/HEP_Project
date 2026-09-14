import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from GAPS_Project.src.data_parse import export_graph_cache_cnndnn_fixedgrid as module


class ExportGraphCacheCNNDNNFixedGridTest(unittest.TestCase):
    def make_graph(self, label: int, selected_index: int) -> SimpleNamespace:
        energies = np.array([2.0, 5.0 + label], dtype=np.float32)
        positions = np.array(
            [[0.0, 0.0, 1000.0], [100.0, -100.0, 0.0]], dtype=np.float32
        )
        raw = np.zeros((2, 8), dtype=np.float32)
        raw[:, :3] = positions
        raw[:, 3] = np.log1p(energies)
        raw[:, 6] = [0.0, 1.0]
        raw[:, 7] = [0.0, 2.0 / 16.0]
        return SimpleNamespace(
            x=torch.tensor(raw),
            pos=torch.tensor(positions),
            y=torch.tensor([label]),
            total_energy=torch.tensor([float(energies.sum())]),
            tof_feat=torch.tensor(np.arange(11, dtype=np.float32) + label),
            mc_beta=torch.tensor([0.3 + 0.01 * label]),
            source_file_index=torch.tensor([label]),
            source_root_entry=torch.tensor([100 + selected_index]),
            source_selected_index=torch.tensor([selected_index]),
            num_nodes=2,
        )

    def make_cache(self, cache: Path) -> None:
        cache.mkdir()
        (cache / "_SUCCESS").write_text("ok\n")
        (cache / "node_feature_normalizer.json").write_text(
            json.dumps(
                {
                    "mode": "global_log",
                    "mean": [0.0] * 6,
                    "std": [1.0] * 6,
                }
            )
        )
        splits = {}
        for split in ("train", "val", "test"):
            splits[split] = {
                "events": 4,
                "label_counts": {"0": 2, "1": 2},
            }
            for particle, label in module.PARTICLES:
                path = cache / f"{split}_{particle}_000.pt"
                torch.save(
                    [self.make_graph(label, index) for index in range(2)], path
                )
                path.with_suffix(".json").write_text(
                    json.dumps({"n_graphs": 2})
                )
        (cache / "cache_manifest.json").write_text(
            json.dumps(
                {
                    "selection": "stopped-toptrigger",
                    "splits": splits,
                }
            )
        )

    def test_exports_same_balanced_split_to_12_by_12_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            output = root / "dataset"
            self.make_cache(cache)

            argv = [
                "export",
                "--cache-dir",
                str(cache),
                "--output-dir",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()

            train = output / "train_cnndnn_10x12x12"
            np.testing.assert_array_equal(
                np.load(train / "labels.npy"), np.array([0, 1, 0, 1])
            )
            voxels = np.load(train / "voxels.npy")
            self.assertEqual(voxels.shape, (4, 10, 12, 12))
            self.assertAlmostEqual(float(voxels[0].sum()), 5.0, places=5)
            self.assertAlmostEqual(float(voxels[1].sum()), 6.0, places=5)
            np.testing.assert_array_equal(
                np.load(train / "source_selected_indices.npy"),
                np.array([0, 0, 1, 1]),
            )
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["event_selection"], "stopped-toptrigger")
            self.assertFalse(manifest["input"]["beta_is_model_input"])
            self.assertTrue((output / "_SUCCESS").is_file())


if __name__ == "__main__":
    unittest.main()

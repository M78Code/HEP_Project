import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from GAPS_Project.src.scripts import build_treemc_fixedgrid_binary_dataset as module


class BuildTreeMcFixedGridBinaryDatasetTest(unittest.TestCase):
    def make_export(self, directory: Path, label: int, count: int) -> None:
        directory.mkdir()
        values = np.arange(count, dtype=np.float32) + 10 * label
        np.save(
            directory / "voxels.npy",
            np.broadcast_to(values[:, None, None, None], (count, 10, 12, 12)).copy(),
        )
        np.save(
            directory / "tof_primary.npy",
            np.broadcast_to(values[:, None], (count, 11)).copy(),
        )
        np.save(directory / "labels.npy", np.full(count, label, dtype=np.int64))
        np.save(directory / "betas.npy", values.astype(np.float32))
        np.save(directory / "random_seeds.npy", np.arange(count, dtype=np.int64))
        np.save(directory / "chain_entries.npy", np.arange(count, dtype=np.int64))
        np.save(
            directory / "source_file_indices.npy", np.zeros(count, dtype=np.int32)
        )
        np.save(directory / "source_entries.npy", np.arange(count, dtype=np.int64))
        (directory / "source_files.txt").write_text(f"label{label}.root\n")
        (directory / "_SUCCESS").write_text("ok\n")

    def test_builds_balanced_interleaved_splits_with_legacy_tof_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            antip = root / "antip"
            antid = root / "antid"
            output = root / "dataset"
            self.make_export(antip, 0, 5)
            self.make_export(antid, 1, 5)

            argv = [
                "build",
                "--antip-dir",
                str(antip),
                "--antid-dir",
                str(antid),
                "--output-dir",
                str(output),
                "--train-per-class",
                "2",
                "--val-per-class",
                "1",
                "--test-per-class",
                "1",
                "--chunk-size",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                module.main()

            train = output / "train_nakagami_style_4M"
            np.testing.assert_array_equal(
                np.load(train / "labels.npy"), np.array([0, 1, 0, 1])
            )
            self.assertEqual(np.load(train / "voxels.npy").shape, (4, 10, 12, 12))
            tof_paddles = np.load(train / "tof_paddles.npy")
            self.assertEqual(tof_paddles.shape, (4, 172))
            self.assertTrue(np.all(tof_paddles == 0))
            self.assertTrue((output / "_SUCCESS").is_file())

            val_beta = np.load(output / "val_nakagami_style_4M" / "betas.npy")
            np.testing.assert_array_equal(val_beta, np.array([2.0, 12.0]))

    def test_rejects_wrong_particle_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "bad"
            self.make_export(directory, 1, 2)
            with self.assertRaisesRegex(ValueError, "expected label 0"):
                module.load_export(directory, 0)


if __name__ == "__main__":
    unittest.main()

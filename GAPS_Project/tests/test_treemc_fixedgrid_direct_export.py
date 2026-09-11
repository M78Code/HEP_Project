import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from GAPS_Project.src.scripts.audit_treemc_fixedgrid_direct_export import (
    compare,
    load_direct_npy,
    load_legacy_csv,
)


class TreeMcFixedGridDirectExportTest(unittest.TestCase):
    def test_direct_arrays_match_legacy_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "events.csv"
            direct_dir = root / "direct"
            direct_dir.mkdir()

            rows = []
            for index, label in enumerate((0, 1)):
                row = ["0"] * 1457
                row[0] = str(1000 + index)
                row[1] = str(2000 + index)
                row[2] = str(label)
                row[4] = str(np.float32(0.25 + index * 0.1))
                row[6 + index] = str(np.float32(1.5 + index))
                row[1446 + index] = str(np.float32(3.5 + index))
                rows.append(row)

            with csv_path.open("w", newline="") as handle:
                csv.writer(handle).writerows(rows)

            legacy = load_legacy_csv(csv_path)
            for name, values in legacy.items():
                np.save(direct_dir / f"{name}.npy", values)
            (direct_dir / "_SUCCESS").touch()

            direct = load_direct_npy(direct_dir)
            compare(legacy, direct)

    def test_compare_rejects_changed_value(self):
        expected = {"values": np.asarray([1.0], dtype=np.float32)}
        actual = {"values": np.asarray([2.0], dtype=np.float32)}
        with self.assertRaisesRegex(AssertionError, "non-equivalent fields"):
            compare(expected, actual)


if __name__ == "__main__":
    unittest.main()

import argparse
import unittest

from GAPS_Project.src.data_parse.build_aohba_fixedgrid_selected_treerec_cache import (
    split_ranges,
)


class SelectedTreeRecCacheSplitTest(unittest.TestCase):
    def test_200k_default_ranges(self):
        args = argparse.Namespace(
            train_events_per_class=80_000,
            val_events_per_class=10_000,
            test_events_per_class=10_000,
        )
        self.assertEqual(
            split_ranges(args),
            {
                "train": (0, 80_000),
                "val": (80_000, 90_000),
                "test": (90_000, 100_000),
            },
        )

    def test_4m_ranges(self):
        args = argparse.Namespace(
            train_events_per_class=1_600_000,
            val_events_per_class=200_000,
            test_events_per_class=200_000,
        )
        self.assertEqual(
            split_ranges(args),
            {
                "train": (0, 1_600_000),
                "val": (1_600_000, 1_800_000),
                "test": (1_800_000, 2_000_000),
            },
        )


if __name__ == "__main__":
    unittest.main()

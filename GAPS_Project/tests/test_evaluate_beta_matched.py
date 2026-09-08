import unittest

from src.scripts.evaluate_beta_matched import format_bin_key


class EvaluateBetaMatchedTest(unittest.TestCase):
    def test_narrow_bin_keys_preserve_edge_precision(self):
        keys = {
            format_bin_key(0.275, 0.280),
            format_bin_key(0.280, 0.285),
            format_bin_key(0.285, 0.290),
        }

        self.assertEqual(
            keys,
            {'0.275-0.28', '0.28-0.285', '0.285-0.29'},
        )


if __name__ == '__main__':
    unittest.main()

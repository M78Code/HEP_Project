import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).parents[1]
    / "tools"
    / "export"
    / "digitize_treemc_skims.py"
)
SPEC = importlib.util.spec_from_file_location("digitize_treemc_skims", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DigitizationSeedTest(unittest.TestCase):
    def test_seed_is_stable(self):
        first = MODULE.derive_digitization_seed(20260906, "sample.root")
        second = MODULE.derive_digitization_seed(20260906, "sample.root")
        self.assertEqual(first, second)
        self.assertGreater(first, 0)
        self.assertLess(first, 2_147_483_647)

    def test_file_name_changes_seed(self):
        first = MODULE.derive_digitization_seed(20260906, "a.root")
        second = MODULE.derive_digitization_seed(20260906, "b.root")
        self.assertNotEqual(first, second)

    def test_seed_base_changes_seed(self):
        first = MODULE.derive_digitization_seed(20260906, "sample.root")
        second = MODULE.derive_digitization_seed(20260907, "sample.root")
        self.assertNotEqual(first, second)

    def test_seed_base_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            MODULE.derive_digitization_seed(0, "sample.root")

    def test_command_contains_explicit_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = MODULE.Job(
                input_path=root / "input.root",
                output_path=root / "output.root",
                log_path=root / "job.log",
                status_path=root / "job.status",
                seed_path=root / "job.seed",
                expected_events=100,
                digitization_seed=123456,
            )
            command = MODULE.build_command(
                SimpleNamespace(crane=Path("/tmp/CraneBaseProcessing")),
                job,
            )
        index = command.index("--digitization-seed")
        self.assertEqual(command[index + 1], "123456")

    def test_command_preserves_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = MODULE.Job(
                input_path=root / "input.root",
                output_path=root / "output.root",
                log_path=root / "job.log",
                status_path=root / "job.status",
                seed_path=root / "job.seed",
                expected_events=100,
                digitization_seed=None,
            )
            command = MODULE.build_command(
                SimpleNamespace(crane=Path("/tmp/CraneBaseProcessing")),
                job,
            )
        self.assertNotIn("--digitization-seed", command)


if __name__ == "__main__":
    unittest.main()

"""Black-box checks for the bounded C verifier."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CheckedRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tempdir.name) / "collatz_range_checked"
        subprocess.run(
            ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
             str(ROOT / "collatz_range_checked.c"), "-o", str(cls.binary)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def run_checker(self, *arguments):
        return subprocess.run(
            [str(self.binary), *map(str, arguments)], text=True,
            capture_output=True, check=False,
        )

    def test_complete_inclusive_sweep(self):
        result = self.run_checker(1, 100000)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["checked"], 100000)
        self.assertTrue(record["complete"])
        self.assertEqual(record["status"], "verified_bounded_range")

    def test_rejects_malformed_and_negative_inputs(self):
        for arguments in (("abc", "5"), ("-5", "5"), ("0", "5"), ("9", "2")):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.run_checker(*arguments).returncode, 64)

    def test_step_limit_is_explicitly_incomplete(self):
        result = self.run_checker(3, 3, 1)
        self.assertEqual(result.returncode, 2)
        record = json.loads(result.stdout)
        self.assertEqual(record["status"], "step_limit")
        self.assertFalse(record["complete"])

    def test_overflow_is_detected_before_arithmetic(self):
        result = self.run_checker(6148914691236517205, 6148914691236517205)
        self.assertEqual(result.returncode, 3)
        record = json.loads(result.stdout)
        self.assertEqual(record["status"], "overflow")
        self.assertFalse(record["complete"])


if __name__ == "__main__":
    unittest.main()

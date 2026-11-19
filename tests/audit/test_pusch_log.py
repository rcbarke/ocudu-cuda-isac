"""Synthetic log fixtures exercise the fail-closed transcript gate."""

import unittest

from scripts.verify_pusch_log import verify_transcript


EXAMPLE = """=== Results (3 iterations, 1 warmup skipped) ===
CPU: 3/3 pass (0.0% BLER)
GPU: 3/3 pass (0.0% BLER)  [resident GPU timing: 3/3]
GPU producer-owned grid: 3/3 measured iterations
GPU immutable owned snapshot: 3/3 measured iterations
Disagreements: 0
Byte mismatches: 0
Output guard failures: 0
PASS: Both paths decoded all 3 iterations, 0 byte mismatches
"""


class VerifyPuschLogTest(unittest.TestCase):
    def test_high_sinr_reference(self):
        self.assertEqual(verify_transcript(EXAMPLE, 3), [])

    def test_each_silent_failure_rejects(self):
        for original, changed in [
            ("immutable owned snapshot: 3/3", "immutable owned snapshot: 2/3"),
            ("resident GPU timing: 3/3", "resident GPU timing: 0/3"),
            ("Disagreements: 0", "Disagreements: 1"),
            ("Byte mismatches: 0", "Byte mismatches: 1"),
            ("Output guard failures: 0", "Output guard failures: 1"),
            ("CPU: 3/3", "CPU: 0/3"),
        ]:
            with self.subTest(original=original):
                self.assertTrue(verify_transcript(EXAMPLE.replace(original, changed), 3))

    def test_incomplete_concatenated_or_inconsistent_run_rejects(self):
        for log in ["", EXAMPLE + EXAMPLE, EXAMPLE + "*** FAIL: output corruption\n",
                    EXAMPLE + "Output guard failures: 2\n",
                    EXAMPLE.split("PASS:")[0]]:
            with self.subTest(log=log):
                self.assertTrue(verify_transcript(log, 3))
        self.assertTrue(verify_transcript(EXAMPLE, 2))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from padtf.audit import pad_timing_consistent
from padtf.runner import ANALYSIS_POLICIES, PAPER_METHODS, TIME_MATCHED_METHODS


class AuditConstantsTests(unittest.TestCase):
    def test_protocol_sets_have_unique_names(self) -> None:
        self.assertEqual(len(PAPER_METHODS), len(set(PAPER_METHODS)))
        self.assertIn("Fixed-FT", PAPER_METHODS)
        self.assertEqual(len(TIME_MATCHED_METHODS), len(set(TIME_MATCHED_METHODS)))
        self.assertEqual(len(ANALYSIS_POLICIES), len(set(ANALYSIS_POLICIES)))

    def test_pad_timing_requires_all_target_components(self) -> None:
        record = {
            "timing": {
                "static": 1.0,
                "zero_shot": 2.0,
                "probe_training": 3.0,
                "grid": 4.0,
                "selection": 5.0,
                "adaptation": 6.0,
                "total_target": 21.0,
            }
        }
        self.assertTrue(pad_timing_consistent(record))
        record["timing"]["total_target"] = 20.0
        self.assertFalse(pad_timing_consistent(record))


if __name__ == "__main__":
    unittest.main()

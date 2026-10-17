from __future__ import annotations

import unittest

from padtf.utility import (
    deployment_utility,
    estimated_utility,
    normalized_target_cost,
    relative_error_reduction,
)


class UtilityTests(unittest.TestCase):
    def test_relative_error_reduction(self) -> None:
        self.assertAlmostEqual(relative_error_reduction(10.0, 7.5), 0.25)

    def test_normalized_target_cost(self) -> None:
        self.assertAlmostEqual(normalized_target_cost(25.0, 100.0), 0.25)
        self.assertEqual(normalized_target_cost(150.0, 100.0, clip=True), 1.0)

    def test_estimated_and_deployment_utility_agree(self) -> None:
        expected = estimated_utility(0.25, 0.2, 0.3)
        actual = deployment_utility(10.0, 7.5, 20.0, 100.0, 0.3)
        self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(actual, 0.19)


if __name__ == "__main__":
    unittest.main()

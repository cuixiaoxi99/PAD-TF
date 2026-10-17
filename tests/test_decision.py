from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from padtf.config import ExperimentConfig
from padtf.decision import DomainSignals, choose_strategy
from padtf.selectors import _nested_coreset_coverage


class DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig()

    def signals(
        self,
        probe_gain: float,
        phi: float = 0.6,
        phi_static: float = 0.2,
    ) -> DomainSignals:
        return DomainSignals(
            delta_nodes=0.2,
            delta_mean=0.2,
            delta_std=0.2,
            delta_graph=0.2,
            phi_static=phi_static,
            zero_shot_mae=10.0,
            naive_mae=10.0,
            degradation=1.0,
            phi=phi,
            probe_gain=probe_gain,
        )

    def test_negative_probe_never_selects_finetune(self) -> None:
        decision = choose_strategy(self.config, "P1", self.signals(-0.1), 10.0, 500.0, 5.0)
        self.assertNotEqual(decision.strategy, "Fine-tune")
        self.assertEqual(decision.candidates, [])

    def test_positive_probe_selects_valid_grid_point(self) -> None:
        decision = choose_strategy(self.config, "P1", self.signals(0.5), 10.0, 500.0, 5.0)
        if decision.strategy == "Fine-tune":
            self.assertIn(decision.ratio, self.config.section("padtf")["ratios"])
            self.assertIn(decision.epochs, self.config.section("padtf")["epochs"])
            best = max(decision.candidates, key=lambda candidate: candidate["utility"])
            self.assertEqual(decision.ratio, best["ratio"])
            self.assertEqual(decision.epochs, best["epochs"])
            self.assertAlmostEqual(decision.utility, best["utility"])

    def test_static_variant_can_use_fixed_response_rate(self) -> None:
        decision = choose_strategy(
            self.config,
            "P1",
            self.signals(-0.1),
            10.0,
            500.0,
            5.0,
            signal_mode="static",
            fixed_epoch_rate=0.1,
        )
        self.assertTrue(decision.candidates)

    def test_no_cost_variant_never_penalizes_time(self) -> None:
        decision = choose_strategy(
            self.config,
            "P1",
            self.signals(0.5),
            10.0,
            500.0,
            5.0,
            cost_weight=0.0,
        )
        self.assertTrue(all(candidate["utility"] == candidate["benefit"] for candidate in decision.candidates))

    def test_static_discrepancy_changes_ratio_response_rate(self) -> None:
        low_shift = choose_strategy(
            self.config,
            "P1",
            self.signals(0.5, phi_static=0.2),
            10.0,
            500.0,
            5.0,
        )
        high_shift = choose_strategy(
            self.config,
            "P1",
            self.signals(0.5, phi_static=0.8),
            10.0,
            500.0,
            5.0,
        )
        self.assertGreater(
            low_shift.candidates[0]["effective_ratio_rate"],
            high_shift.candidates[0]["effective_ratio_rate"],
        )

    def test_ratio_time_model_includes_fixed_epoch_overhead(self) -> None:
        decision = choose_strategy(
            self.config,
            "P1",
            self.signals(0.5),
            10.0,
            500.0,
            5.0,
        )
        smallest = min(decision.candidates, key=lambda candidate: candidate["ratio"])
        self.assertGreater(smallest["epoch_time_scale"], smallest["ratio"])
        self.assertTrue(
            all(
                candidate["cost"] <= self.config.section("padtf")["max_candidate_cost"]
                for candidate in decision.candidates
            )
        )

    def test_pair_specific_coverage_changes_ratio_benefit(self) -> None:
        coverage = {0.05: 0.10, 0.1: 0.25, 0.2: 0.55, 0.3: 0.75, 0.5: 0.9, 0.7: 0.97, 1.0: 1.0}
        decision = choose_strategy(
            self.config,
            "P1",
            self.signals(0.5),
            10.0,
            500.0,
            5.0,
            ratio_coverage=coverage,
        )
        candidate = next(item for item in decision.candidates if item["ratio"] == 0.2 and item["epochs"] == 2)
        self.assertNotEqual(candidate["parametric_coverage"], candidate["observed_coverage"])
        self.assertGreater(candidate["mixed_coverage"], 0.0)
        self.assertLessEqual(candidate["mixed_coverage"], 1.0)

    def test_nested_coverage_is_monotonic(self) -> None:
        features = np.asarray([[0.0], [1.0], [2.0], [4.0]], dtype=np.float32)
        coverage = _nested_coreset_coverage(features, [0.25, 0.5, 1.0])
        self.assertLessEqual(coverage[0.25], coverage[0.5])
        self.assertLessEqual(coverage[0.5], coverage[1.0])
        self.assertAlmostEqual(coverage[1.0], 1.0)

    def test_graph_expansion_rebalances_data_and_epochs(self) -> None:
        coverage = {0.05: 0.25, 0.1: 0.4, 0.2: 0.55, 0.3: 0.65, 0.5: 0.8, 0.7: 0.9, 1.0: 1.0}
        decision = choose_strategy(
            self.config,
            "P1",
            self.signals(0.5),
            10.0,
            500.0,
            5.0,
            epoch_candidates=[45],
            ratio_coverage=coverage,
            signed_log_nodes=1.4,
            flow_target=True,
        )
        adjusted = [candidate for candidate in decision.candidates if candidate["structural_adjustment"]]
        self.assertEqual(len(adjusted), 1)
        self.assertGreater(adjusted[0]["ratio"], 0.1)
        self.assertLess(adjusted[0]["epochs"], 45)
        self.assertEqual(decision.ratio, adjusted[0]["ratio"])
        self.assertEqual(decision.epochs, adjusted[0]["epochs"])


if __name__ == "__main__":
    unittest.main()

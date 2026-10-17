from __future__ import annotations

import unittest

import numpy as np

from padtf.data import (
    StandardScaler,
    WindowDataset,
    blocked_stratified_probe_split,
    split_indices,
    stratified_indices,
)


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        values = np.arange(80 * 4, dtype=np.float32).reshape(80, 4)
        self.dataset = WindowDataset(values, StandardScaler(10.0, 2.0), 0, 80, 12, 12, 1, 5)

    def test_window_shapes(self) -> None:
        inputs, targets = self.dataset[0]
        self.assertEqual(tuple(inputs.shape), (12, 4, 3))
        self.assertEqual(tuple(targets.shape), (12, 4, 1))

    def test_probe_split_is_disjoint(self) -> None:
        indices = stratified_indices(self.dataset, 0.2, 2024)
        train, val = split_indices(indices, 0.8, 2024)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(indices))

    def test_blocked_probe_split_has_no_window_overlap(self) -> None:
        train, val = blocked_stratified_probe_split(self.dataset, 0.2, 0.8, 2024, 24)
        self.assertFalse(set(train) & set(val))
        self.assertGreaterEqual(min(val) - max(train), 24)
        self.assertEqual(len(train) + len(val), int(np.ceil(len(self.dataset) * 0.2)))

    def test_blocked_probe_gap_covers_history_and_horizon(self) -> None:
        with self.assertRaises(ValueError):
            blocked_stratified_probe_split(self.dataset, 0.2, 0.8, 2024, 23)


if __name__ == "__main__":
    unittest.main()

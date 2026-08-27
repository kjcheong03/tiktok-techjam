from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ghostlab.retrieval.gbdt import LambdaMARTModel
from ghostlab.retrieval.reward_lambdamart import (
    fit_reward_lambdamart,
    mean_predicted_terminal_reward,
    ranking_derivatives,
)


class RewardLambdaMARTTests(unittest.TestCase):
    def test_turn_aware_boundary_weight_is_larger_early(self) -> None:
        labels = np.asarray([0] * 9 + [1, 0], dtype=np.int64)
        scores = np.arange(11, 0, -1, dtype=np.float64)
        scores[9] = 0.0
        scores[10] = -1.0
        early, _ = ranking_derivatives(
            labels, scores, [slice(0, 11)], [1], objective="turn_aware_reward"
        )
        late, _ = ranking_derivatives(
            labels, scores, [slice(0, 11)], [10], objective="turn_aware_reward"
        )
        self.assertGreater(abs(early[9]), abs(late[9]))

    def test_mean_predicted_reward_uses_stable_target_rank(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        scores = np.asarray([0.2, 0.9, 0.1, 0.0], dtype=np.float64)
        value = mean_predicted_terminal_reward(labels, scores, [2, 2], [1, 10])
        self.assertAlmostEqual(value, (1.0 + 0.67) / 2)

    def test_fit_is_deterministic_compact_and_target_free_at_runtime(self) -> None:
        features = np.asarray(
            [[0.0], [1.0], [0.0], [1.0]] * 20, dtype=np.float64
        )
        labels = np.asarray([0, 1, 0, 1] * 20, dtype=np.int64)
        groups = [2] * 40
        turns = [1, 10] * 20
        kwargs = {
            "candidate_id": "toy_reward",
            "feature_names": ("observable_overlap",),
            "objective": "turn_aware_reward",
            "max_depth": 1,
            "num_leaves": 2,
            "learning_rate": 0.1,
            "max_rounds": 3,
            "early_stopping_rounds": 2,
            "min_samples_leaf": 1,
        }
        first = fit_reward_lambdamart(features, labels, groups, turns, **kwargs)
        second = fit_reward_lambdamart(features, labels, groups, turns, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first.candidate_id, "toy_reward")
        self.assertNotIn("target", first.feature_names)
        self.assertNotIn("turn", first.feature_names)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.json"
            first.save(path)
            loaded = LambdaMARTModel.load(path)
        self.assertEqual(first, loaded)


if __name__ == "__main__":
    unittest.main()

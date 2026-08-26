from __future__ import annotations

import unittest

from ghostlab.research.route_policy import RouteFeatures, fit_route_table


class RoutePolicyTest(unittest.TestCase):
    def test_route_table_uses_training_rewards(self) -> None:
        features = {f"s{index}": RouteFeatures(index < 5, False) for index in range(10)}
        rewards = {
            "keyword": {
                key: float(value.has_initial_constraint)
                for key, value in features.items()
            },
            "dense": {
                key: float(not value.has_initial_constraint)
                for key, value in features.items()
            },
        }
        table = fit_route_table(
            features,
            rewards,
            features,
            ("keyword", "dense"),
            ("has_initial_constraint",),
        )
        self.assertEqual(table.predict(features["s0"]), "keyword")
        self.assertEqual(table.predict(features["s9"]), "dense")


if __name__ == "__main__":
    unittest.main()

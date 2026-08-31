from __future__ import annotations

import unittest

from ghostlab.research.route_stump import fit_route_stump


class RouteStumpTests(unittest.TestCase):
    def test_finds_general_route_split(self) -> None:
        sample_ids = [str(index) for index in range(40)]
        features = {
            sample_id: {"margin": float(index)}
            for index, sample_id in enumerate(sample_ids)
        }
        rewards = {
            "dense": {
                sample_id: float(index < 20)
                for index, sample_id in enumerate(sample_ids)
            },
            "keyword": {
                sample_id: float(index >= 20)
                for index, sample_id in enumerate(sample_ids)
            },
        }
        stump = fit_route_stump(
            sample_ids,
            rewards,
            features,
            ("keyword", "dense"),
            minimum_leaf_sessions=10,
        )
        self.assertEqual(stump.predict({"margin": 2.0}), "dense")
        self.assertEqual(stump.predict({"margin": 35.0}), "keyword")

    def test_prefers_constant_route_inside_tie_band(self) -> None:
        sample_ids = [str(index) for index in range(30)]
        features = {
            sample_id: {"x": float(index)} for index, sample_id in enumerate(sample_ids)
        }
        rewards = {
            "keyword": {sample_id: 0.7 for sample_id in sample_ids},
            "dense": {sample_id: 0.695 for sample_id in sample_ids},
        }
        stump = fit_route_stump(
            sample_ids,
            rewards,
            features,
            ("keyword", "dense"),
            minimum_leaf_sessions=5,
        )
        self.assertIsNone(stump.feature)
        self.assertEqual(stump.default_route, "keyword")


if __name__ == "__main__":
    unittest.main()

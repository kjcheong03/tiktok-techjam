from __future__ import annotations

import unittest

from ghostlab.research.firewall import reject_forbidden_names, runtime_profile
from ghostlab.research.replay import ReplayEnvironment


class ReplaySafetyTest(unittest.TestCase):
    def test_runtime_profile_does_not_expose_research_fields(self) -> None:
        sample = {
            "user_profile": {"summary": "safe"},
            "ground_truth": {"parent_asin": "x"},
        }
        self.assertEqual(runtime_profile(sample), {"summary": "safe"})
        with self.assertRaises(ValueError):
            reject_forbidden_names(["turn", "ground_truth"])

    def test_snapshot_clone_is_independent(self) -> None:
        sample = {
            "sample_id": "s",
            "scenario_type": "browsing",
            "ground_truth": {"parent_asin": "target"},
            "user_profile": {},
        }
        categories = {"target": ["Clothing", "Shoes"]}
        products = {"target": {"parent_asin": "target", "title": "Trail Shoe"}}
        environment = ReplayEnvironment(sample, categories, products)
        clone = environment.clone()
        clone.step({"ask_attribute": "material", "recommendations": []})
        self.assertEqual(environment.turn, 1)
        self.assertEqual(clone.turn, 2)

    def test_restore_round_trip(self) -> None:
        sample = {
            "sample_id": "s",
            "scenario_type": "boundary",
            "ground_truth": {"parent_asin": "target"},
            "user_profile": {},
        }
        environment = ReplayEnvironment(
            sample,
            {"target": ["Clothing", "Shirts"]},
            {"target": {"parent_asin": "target", "title": "Shirt"}},
        )
        snapshot = environment.snapshot()
        environment.step({"ask_attribute": "color", "recommendations": []})
        environment.restore(snapshot)
        self.assertEqual(environment.snapshot(), snapshot)


if __name__ == "__main__":
    unittest.main()

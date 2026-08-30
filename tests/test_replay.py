from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluator.local_evaluator import evaluate
from ghostlab.research.firewall import reject_forbidden_names, runtime_profile
from ghostlab.research.replay import ReplayEnvironment, evaluate_shared


class _FixedAgent:
    def __init__(self, response: object) -> None:
        self.response = response

    def reset(self, session_id: str, user_profile: dict | None = None) -> None:
        del session_id, user_profile

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> object:
        del session_id, user_message, turn, top_k
        return self.response


class ReplaySafetyTest(unittest.TestCase):
    def test_shared_harness_matches_published_evaluator(self) -> None:
        sample = {
            "sample_id": "parity-sample",
            "scenario_type": "buying",
            "ground_truth": {"parent_asin": "target"},
            "user_profile": {"summary": "safe"},
            "intent_card": {
                "target_category": "Trail Shoe",
                "hard_constraints": ["waterproof"],
                "soft_preferences": ["blue"],
            },
            "behavior": {"scenario_type": "buying"},
        }
        categories = {"target": ["Clothing", "Shoes"]}
        products = {"target": {"parent_asin": "target", "title": "Trail Shoe"}}
        with TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                '{"parent_asin":"target","title":"Trail Shoe"}\n',
                encoding="utf-8",
            )
            published = evaluate(
                _FixedAgent({"message": "found", "recommendations": ["target"]}),
                [sample],
                {"target"},
                categories,
                products,
            )
            shared = evaluate_shared(
                _FixedAgent({"message": "found", "recommendations": ["target"]}),
                [sample],
                categories,
                products,
                catalog_path=catalog,
            )
        shared.pop("evaluation_contract")
        self.assertEqual(shared, published)

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

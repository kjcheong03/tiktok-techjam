from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.policy.calibrated_router import (
    CalibratedRouteModel,
    RouterTrainingState,
    fit_calibrated_router,
)
from ghostlab.runtime.unified_experimental import ExperimentalAgent


class CalibratedRouterTests(unittest.TestCase):
    def test_router_learns_positive_region_and_retains_base(self) -> None:
        states = [
            RouterTrainingState(
                str(index),
                {"uncertainty": index / 39},
                {
                    "keyword": 0.6,
                    "dense": 0.9 if index >= 20 else 0.3,
                },
            )
            for index in range(40)
        ]
        model = fit_calibrated_router(
            states[:30],
            states[30:],
            feature_names=("uncertainty",),
            routes=("keyword", "dense"),
            minimum_routed_sessions=3,
        )
        self.assertEqual(model.decide({"uncertainty": 0.1}).route, "keyword")
        self.assertEqual(model.decide({"uncertainty": 0.9}).route, "dense")
        self.assertGreaterEqual(model.calibration_precision, 0.6)

    def test_forbidden_feature_is_rejected(self) -> None:
        state = RouterTrainingState(
            "s", {"target": 1.0}, {"keyword": 0.1, "dense": 0.2}
        )
        with self.assertRaisesRegex(ValueError, "runtime boundary"):
            fit_calibrated_router(
                [state],
                [state],
                feature_names=("target",),
                routes=("keyword", "dense"),
                minimum_routed_sessions=1,
            )

    def test_runtime_switch_is_explicit_and_base_route_is_safe(self) -> None:
        feature_names = (
            "turn_fraction",
            "active_slot_fraction",
            "asked_fraction",
            "no_preference_fraction",
            "previous_candidate_fraction",
            "previous_margin",
            "previous_entropy",
            "category_known",
        )
        model = CalibratedRouteModel(
            feature_names,
            "keyword",
            {"keyword": (0.0,) * (len(feature_names) + 1)},
            0.0,
            1.0,
            10,
            5,
        )
        product = {
            "parent_asin": "A",
            "title": "shoe",
            "categories": ["Shoes"],
            "features": [],
            "description": [],
            "details": {},
            "store": "One",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps(product) + "\n", encoding="utf-8")
            agent = ExperimentalAgent(
                path,
                state_variant="raw_history",
                routing_variant="calibrated",
                calibrated_router=model,
            )
            agent.reset("s", {})
            agent.respond("s", "I want shoes", 1, 10)
        self.assertEqual(agent.retrieval_trace[-1]["route"], "keyword")
        self.assertEqual(
            agent.retrieval_trace[-1]["route_decision"]["reason"], "always_base"
        )


if __name__ == "__main__":
    unittest.main()

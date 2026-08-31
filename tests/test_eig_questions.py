from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.policy.candidate_statistics import (
    CandidateStatistics,
    FacetDistribution,
)
from ghostlab.policy.eig_questions import CandidateEIGPolicy
from ghostlab.runtime.unified_experimental import ExperimentalAgent
from ghostlab.state.memory import ConversationState


def facet(attribute: str, gain: float, coverage: int = 100) -> FacetDistribution:
    return FacetDistribution(
        attribute=attribute,  # type: ignore[arg-type]
        counts=(("a", 50), ("b", 50)),
        candidate_count=100,
        covered_count=coverage,
        entropy=0.69,
        normalized_entropy=gain,
        partition_gain=gain,
        expected_reduction=100 * gain,
        no_preference_probability=1 - coverage / 100,
    )


class EIGQuestionTests(unittest.TestCase):
    def test_selects_best_legal_candidate_partition(self) -> None:
        state = ConversationState("s", {})
        statistics = CandidateStatistics(
            100,
            {"color": facet("color", 0.8), "size": facet("size", 0.4)},
        )
        decision = CandidateEIGPolicy().decide(
            state, statistics, turn=3, message="still exploring"
        )
        self.assertEqual(decision.ask_attribute, "color")
        self.assertEqual(decision.reason, "candidate_information_gain")

    def test_known_and_declined_actions_are_never_selected(self) -> None:
        state = ConversationState("s", {})
        state.no_preference_attributes.add("color")
        decision = CandidateEIGPolicy().decide(
            state,
            CandidateStatistics(100, {"color": facet("color", 0.9)}),
            turn=2,
            message="no color preference",
        )
        self.assertNotEqual(decision.ask_attribute, "color")

    def test_broad_discovery_is_deliberate_and_bounded(self) -> None:
        decision = CandidateEIGPolicy().decide(
            ConversationState("s", {}),
            CandidateStatistics(100, {"color": facet("color", 0.9)}),
            turn=1,
            message="still exploring",
        )
        self.assertEqual(decision.ask_attribute, "other")
        self.assertEqual(decision.reason, "broad_discovery")

    def test_low_information_stops(self) -> None:
        decision = CandidateEIGPolicy(question_value_margin=0.2).decide(
            ConversationState("s", {}),
            CandidateStatistics(100, {"color": facet("color", 0.01)}),
            turn=3,
            message="specific request",
        )
        self.assertIsNone(decision.ask_attribute)
        self.assertEqual(decision.reason, "reward_aware_stop")

    def test_runtime_switch_is_off_by_default_and_enabled_is_legal(self) -> None:
        rows = [
            {
                "parent_asin": "A",
                "title": "red shoe",
                "categories": ["Shoes"],
                "features": [],
                "description": [],
                "details": {"Color": "red"},
                "store": "One",
            },
            {
                "parent_asin": "B",
                "title": "blue shoe",
                "categories": ["Shoes"],
                "features": [],
                "description": [],
                "details": {"Color": "blue"},
                "store": "Two",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            control = ExperimentalAgent(path)
            enabled = ExperimentalAgent(path, question_variant="candidate_eig")
            self.assertIsNone(control.candidate_facets)
            control.reset("c", {})
            enabled.reset("e", {})
            response = enabled.respond("e", "I want shoes", 1, 10)
        self.assertEqual(response["ask_attribute"], "other")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.retrieval.constraint_gbdt import (
    ConstraintContext,
    ConstraintGBDTFeatureStore,
)
from ghostlab.state.memory import ConversationState


class ConstraintGBDTTests(unittest.TestCase):
    def _catalog(self, directory: str) -> Path:
        path = Path(directory) / "catalog.jsonl"
        rows = [
            {
                "parent_asin": "wool",
                "title": "warm wool coat",
                "features": ["red winter outerwear"],
            },
            {"parent_asin": "cotton", "title": "blue cotton shirt"},
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    @staticmethod
    def _state() -> ConversationState:
        state = ConversationState("s", {}, negative_evidence=True)
        state.observe("I'm looking for a coat. What I need is: warm wool.", 1)
        state.last_asked_attribute = "color"
        state.asked_attributes.append("color")
        state.observe("For that, what matters is: red.", 2)
        return state

    def test_active_constraint_coverage_and_provenance_are_candidate_specific(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConstraintGBDTFeatureStore(self._catalog(directory))
            context = ConstraintContext.from_runtime(
                self._state(), turn=2, retrieval_scores=[4.0, 2.0]
            )
            wool = store.constraint_features("wool", context)
            cotton = store.constraint_features("cotton", context)
        self.assertGreater(
            wool["active_constraint_coverage_ratio"],
            cotton["active_constraint_coverage_ratio"],
        )
        self.assertGreater(wool["explicit_constraint_coverage_ratio"], 0.0)
        self.assertGreater(wool["simulator_constraint_coverage_ratio"], 0.0)

    def test_negative_values_are_contradictions_but_no_preference_is_neutral(
        self,
    ) -> None:
        state = self._state()
        state.observe("Please avoid wool.", 3)
        state.observe("I don't have a preference for size.", 4)
        with tempfile.TemporaryDirectory() as directory:
            store = ConstraintGBDTFeatureStore(self._catalog(directory))
            context = ConstraintContext.from_runtime(
                state, turn=4, retrieval_scores=[3.0, 1.0]
            )
            wool = store.constraint_features("wool", context)
            cotton = store.constraint_features("cotton", context)
        self.assertGreater(wool["negative_contradiction_count"], 0.0)
        self.assertEqual(cotton["negative_contradiction_count"], 0.0)
        self.assertEqual(wool["no_preference_count"], 1.0)
        self.assertEqual(wool["no_preference_count"], cotton["no_preference_count"])

    def test_invalidated_values_are_not_used_for_positive_coverage(self) -> None:
        state = self._state()
        state.observe("Actually, I'm looking for a shirt instead.", 3)
        with tempfile.TemporaryDirectory() as directory:
            store = ConstraintGBDTFeatureStore(self._catalog(directory))
            context = ConstraintContext.from_runtime(
                state, turn=3, retrieval_scores=[2.0, 1.0]
            )
            wool = store.constraint_features("wool", context)
        self.assertGreater(wool["invalidated_constraint_count"], 0.0)
        self.assertEqual(wool["override_invalidation_present"], 1.0)
        self.assertEqual(
            [item.terms for item in context.positive if "wool" in item.terms], []
        )

    def test_contextual_matrix_rejects_unknown_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConstraintGBDTFeatureStore(self._catalog(directory))
            context = ConstraintContext.from_runtime(
                self._state(), turn=2, retrieval_scores=[]
            )
            with self.assertRaisesRegex(ValueError, "unknown constraint GBDT"):
                store.contextual_matrix("coat", ["wool"], context, ("target",))


if __name__ == "__main__":
    unittest.main()

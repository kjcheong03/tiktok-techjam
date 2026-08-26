from __future__ import annotations

import unittest

from ghostlab.policy.adaptive_questions import (
    AdaptiveQuestionPolicy,
    QuestionContext,
)
from ghostlab.policy.signals import RetrievalSignals


def context(turn: int, **overrides: object) -> QuestionContext:
    values: dict[str, object] = {
        "turn": turn,
        "message": "Still looking.",
        "active_attributes": frozenset(),
        "asked_attributes": frozenset(),
        "no_preference_attributes": frozenset(),
        "last_asked_attribute": None,
        "retrieval": None,
    }
    values.update(overrides)
    return QuestionContext(**values)  # type: ignore[arg-type]


class AdaptiveQuestionPolicyTests(unittest.TestCase):
    def test_starts_with_two_broad_discovery_questions(self) -> None:
        policy = AdaptiveQuestionPolicy()
        self.assertEqual(policy.decide(context(1)).ask_attribute, "other")
        self.assertEqual(policy.decide(context(2)).ask_attribute, "other")

    def test_asks_highest_priority_missing_constraint(self) -> None:
        decision = AdaptiveQuestionPolicy().decide(
            context(3, active_attributes=frozenset({"category", "budget"}))
        )
        self.assertEqual(decision.ask_attribute, "color")
        self.assertEqual(decision.reason, "missing_constraint")

    def test_unhelpful_specific_answer_returns_to_discovery(self) -> None:
        decision = AdaptiveQuestionPolicy().decide(
            context(
                4,
                message="I don't have an additional preference for color.",
                last_asked_attribute="color",
            )
        )
        self.assertEqual(decision.ask_attribute, "other")
        self.assertEqual(decision.reason, "recover_unhelpful_specific")

    def test_low_entropy_can_stop_questions(self) -> None:
        decision = AdaptiveQuestionPolicy().decide(
            context(
                4,
                active_attributes=frozenset({"category", "budget", "color"}),
                retrieval=RetrievalSignals(200, 0.8, 0.1, None),
            )
        )
        self.assertIsNone(decision.ask_attribute)
        self.assertEqual(decision.reason, "confident_stop")

    def test_question_budget_is_bounded(self) -> None:
        decision = AdaptiveQuestionPolicy().decide(context(10))
        self.assertIsNone(decision.ask_attribute)
        self.assertEqual(decision.reason, "question_budget_exhausted")


if __name__ == "__main__":
    unittest.main()

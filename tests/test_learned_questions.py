from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.policy.learned_questions import (
    ACTION_ORDER,
    FEATURE_NAMES,
    LinearActionValueModel,
    QuestionTrainingState,
    fit_linear_action_value,
    legal_question_actions,
    observable_question_features,
)
from ghostlab.runtime.experimental_questions import ExperimentalAgent
from ghostlab.state.memory import ConversationState


def blank_features(**updates: float) -> dict[str, float]:
    result = dict.fromkeys(FEATURE_NAMES, 0.0)
    result.update(updates)
    return result


class LearnedQuestionPolicyTests(unittest.TestCase):
    def test_legal_actions_exclude_known_asked_and_declined_slots(self) -> None:
        state = ConversationState("session", {})
        state.observe("I'm looking for shirts. A key requirement is: cotton.", 1)
        state.asked_attributes.append("color")
        state.no_preference_attributes.add("size")
        actions = legal_question_actions(state)
        self.assertNotIn("category", actions)
        self.assertNotIn("material", actions)
        self.assertNotIn("color", actions)
        self.assertNotIn("size", actions)
        self.assertIn("other", actions)
        self.assertIn(None, actions)

    def test_features_are_observable_and_schema_stable(self) -> None:
        state = ConversationState("session", {})
        message = "I'm looking for shirts, but I'm still exploring."
        state.observe(message, 1)
        features = observable_question_features(
            state,
            message=message,
            query=message,
            turn=1,
            retrieval_scores=[3.0, 2.0, 1.0],
        )
        self.assertEqual(tuple(features), FEATURE_NAMES)
        forbidden = {"target", "rank", "scenario", "reward", "future_answer"}
        self.assertFalse(any(term in name for term in forbidden for name in features))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features.values()))

    def test_linear_fit_is_deterministic_and_respects_legal_actions(self) -> None:
        states = [
            QuestionTrainingState(
                "a",
                1,
                blank_features(turn_fraction=0.1),
                {"other": 0.7, None: 0.2},
            ),
            QuestionTrainingState(
                "b",
                2,
                blank_features(turn_fraction=0.2),
                {"other": 0.6, None: 0.3},
            ),
        ]
        first = fit_linear_action_value(states)
        second = fit_linear_action_value(states)
        self.assertEqual(first, second)
        action, values = first.decide(states[0].features, ("other", None))
        self.assertEqual(action, "other")
        self.assertEqual(set(values), {"other", None})

    def test_stop_is_absorbing_in_runtime(self) -> None:
        width = len(FEATURE_NAMES) + 1
        weights = {action: [0.0] * width for action in ACTION_ORDER}
        weights["other"][0] = 0.25
        weights[None][0] = 1.0
        weights[None][1 + FEATURE_NAMES.index("turn_fraction")] = -5.0
        model = LinearActionValueModel(
            FEATURE_NAMES,
            {action: tuple(value) for action, value in weights.items()},
            1.0,
            2,
        )
        product = {
            "parent_asin": "A",
            "title": "cotton shirt",
            "categories": ["shirts"],
            "features": ["cotton"],
            "details": {},
            "store": "shop",
            "description": "shirt",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps(product) + "\n", encoding="utf-8")
            agent = ExperimentalAgent(
                path,
                state_variant="raw_history",
                question_variant="learned",
                learned_question_model=model,
                sparse_weights=(2.0, 8.0, 4.0, 2.5, 1.5, 1.0),
            )
            agent.reset("session", {})
            first = agent.respond("session", "I'm looking for shirts.", 1, 10)
            second = agent.respond("session", "For that, cotton.", 2, 10)
        self.assertIsNone(first["ask_attribute"])
        self.assertIsNone(second["ask_attribute"])
        self.assertEqual(agent.question_trace[-1]["reason"], "absorbing_stop")


if __name__ == "__main__":
    unittest.main()

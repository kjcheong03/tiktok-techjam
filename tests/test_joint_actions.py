from __future__ import annotations

import unittest

from ghostlab.policy.joint_actions import (
    JOINT_FEATURE_NAMES,
    legalize_joint_action,
    observable_joint_features,
)
from ghostlab.policy.models import JointAction
from ghostlab.state.memory import ConversationState


class JointActionTests(unittest.TestCase):
    def test_features_are_observable_and_bounded(self) -> None:
        state = ConversationState("s", {})
        state.observe("I'm looking for shoes.", 1)
        features = observable_joint_features(
            state, turn=2, previous_scores=[2.0, 1.0, 0.5]
        )
        self.assertEqual(tuple(features), JOINT_FEATURE_NAMES)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features.values()))
        self.assertFalse(any("target" in name or "reward" in name for name in features))

    def test_unregistered_route_and_illegal_question_fail_closed(self) -> None:
        state = ConversationState("s", {})
        state.no_preference_attributes.add("color")
        base = JointAction(ask_attribute=None, retrieval_route="keyword", retrieval_k=100)
        selected = JointAction(
            ask_attribute="color", retrieval_route="dense", retrieval_k=200
        )
        action = legalize_joint_action(
            selected,
            state,
            allowed_routes=frozenset({"keyword"}),
            allowed_depths=frozenset({100}),
            base_action=base,
        )
        self.assertIsNone(action.ask_attribute)
        self.assertEqual(action.retrieval_route, "keyword")
        self.assertEqual(action.retrieval_k, 100)


if __name__ == "__main__":
    unittest.main()

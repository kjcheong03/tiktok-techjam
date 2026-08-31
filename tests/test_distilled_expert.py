from __future__ import annotations

import unittest

from ghostlab.policy.distilled_expert import (
    DistilledExpertPolicy,
    DistilledPolicyModel,
    fit_distilled_policy,
)
from ghostlab.policy.models import JointAction
from ghostlab.research.counterfactual_expert import ExpertLabel
from ghostlab.state.memory import ConversationState


class DistilledExpertTests(unittest.TestCase):
    def labels(self) -> list[ExpertLabel]:
        result = []
        for index in range(40):
            high = index >= 20
            rewards = {
                "base": float(not high),
                "ask": float(high),
            }
            best = "ask" if high else "base"
            result.append(
                ExpertLabel(
                    str(index),
                    1,
                    {"uncertainty": float(index) / 39},
                    rewards,
                    best,
                    1.0,
                    0.0,
                )
            )
        return result

    def test_tree_recovers_general_split_and_round_trips(self) -> None:
        model = fit_distilled_policy(
            self.labels(),
            feature_names=("uncertainty",),
            action_order=("base", "ask"),
            minimum_leaf_sessions=10,
        )
        loaded = DistilledPolicyModel.from_payload(model.to_payload())
        self.assertEqual(loaded.predict({"uncertainty": 0.1})[0], "base")
        self.assertEqual(loaded.predict({"uncertainty": 0.9})[0], "ask")

    def test_runtime_confidence_falls_back_and_legalizes(self) -> None:
        model = fit_distilled_policy(
            self.labels(),
            feature_names=("uncertainty",),
            action_order=("base", "ask"),
            minimum_leaf_sessions=10,
        )
        policy = DistilledExpertPolicy(
            model,
            {
                "base": JointAction(retrieval_route="keyword", retrieval_k=100),
                "ask": JointAction(
                    ask_attribute="color", retrieval_route="keyword", retrieval_k=200
                ),
            },
            frozenset({"keyword"}),
            frozenset({100, 200}),
        )
        state = ConversationState("s", {})
        state.no_preference_attributes.add("color")
        decision = policy.decide(state, {"uncertainty": 0.9})
        self.assertIsNone(decision.action.ask_attribute)


if __name__ == "__main__":
    unittest.main()

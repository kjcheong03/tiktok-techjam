from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.policy.joint_actions import JointTrainingState
from ghostlab.policy.joint_policy import JointObservablePolicy
from ghostlab.policy.models import (
    ActionPatch,
    DecisionList,
    JointAction,
    PolicyRule,
    Predicate,
)
from ghostlab.research.joint_counterfactual import fit_joint_action_table
from ghostlab.runtime.unified_experimental import ExperimentalAgent
from ghostlab.state.memory import ConversationState


class JointPolicyTests(unittest.TestCase):
    def policy(self) -> JointObservablePolicy:
        return JointObservablePolicy(
            DecisionList(
                rules=(
                    PolicyRule(
                        rule_id="early",
                        all_conditions=(
                            Predicate(feature="turn_fraction", operator="le", value=0.2),
                        ),
                        action_patch=ActionPatch(
                            ask_attribute="other", retrieval_k=200
                        ),
                    ),
                ),
                default_action=JointAction(
                    ask_attribute=None, retrieval_route="keyword", retrieval_k=100
                ),
            )
        )

    def test_decision_list_selects_legal_joint_action(self) -> None:
        decision = self.policy().decide(
            ConversationState("s", {}), {"turn_fraction": 0.1}
        )
        self.assertEqual(decision.action.ask_attribute, "other")
        self.assertEqual(decision.action.retrieval_k, 200)

    def test_fold_local_table_recovers_observable_interaction(self) -> None:
        states = [
            JointTrainingState(
                str(index),
                1,
                {"uncertain": float(index < 10)},
                {
                    "ask": float(index < 10),
                    "stop": float(index >= 10),
                },
            )
            for index in range(20)
        ]
        model = fit_joint_action_table(
            states,
            feature_names=("uncertain",),
            action_ids=("ask", "stop"),
            minimum_cell_sessions=5,
        )
        self.assertEqual(model.predict({"uncertain": 1.0}), "ask")
        self.assertEqual(model.predict({"uncertain": 0.0}), "stop")

    def test_runtime_switch_is_explicit_and_records_route(self) -> None:
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
                question_variant="joint_observable",
                joint_policy=self.policy(),
            )
            agent.reset("s", {})
            response = agent.respond("s", "I want shoes", 1, 10)
        self.assertEqual(response["ask_attribute"], "other")
        self.assertEqual(agent.retrieval_trace[-1]["route"], "keyword")


if __name__ == "__main__":
    unittest.main()

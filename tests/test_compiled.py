from __future__ import annotations

import unittest
from pathlib import Path

from ghostlab.policy.models import RuntimeConfig
from ghostlab.runtime.agent import GhostLabRuntime


class CompiledRuntimeTest(unittest.TestCase):
    def test_primary_is_guarded_sequence_and_contract_safe(self) -> None:
        agent = GhostLabRuntime("data/catalog.jsonl")
        agent.reset("session", {})
        attributes = []
        for turn, message in enumerate(
            ("I'm looking for shirts.", "Something comfortable.", "For daily use."),
            start=1,
        ):
            response = agent.respond("session", message, turn, 10)
            attributes.append(response["ask_attribute"])
            self.assertLessEqual(len(response["recommendations"]), 10)
        self.assertEqual(attributes, ["other", "other", "use_case"])

    def test_default_policy_contains_the_guarded_candidate(self) -> None:
        config = RuntimeConfig.model_validate_json(
            Path("configs/compiled_policy.json").read_text(encoding="utf-8")
        )
        techniques = config.techniques
        self.assertEqual(config.policy_id, "ghostlab_guarded_constraint_gbdt_v1")
        self.assertEqual(techniques.state_mode, "raw_history")
        self.assertEqual(techniques.sparse_field_weights, (2, 8, 4, 2.5, 1.5, 1))
        self.assertEqual(techniques.reranker, "guarded_constraint_gbdt")
        assert techniques.base_model_asset is not None
        assert techniques.constraint_model_asset is not None
        self.assertEqual(
            techniques.base_model_asset.sha256,
            "10782d08ce20f8c9a60d3e2482ff577c887a35cc74e456c69c781409eb4df4d6",
        )
        self.assertEqual(
            techniques.constraint_model_asset.sha256,
            "2a3dc13284bb5ca53b9b795c9ec69ac921883be55efe6a239072302c4d4f6e6b",
        )

    def test_two_instances_are_deterministic(self) -> None:
        first = GhostLabRuntime("data/catalog.jsonl")
        second = GhostLabRuntime("data/catalog.jsonl")
        for agent in (first, second):
            agent.reset("session", {})
        messages = ("I'm looking for shirts.", "Something comfortable.")
        for turn, message in enumerate(messages, start=1):
            self.assertEqual(
                first.respond("session", message, turn, 10),
                second.respond("session", message, turn, 10),
            )

    def test_manual_strong_is_still_loadable(self) -> None:
        agent = GhostLabRuntime(
            "data/catalog.jsonl", Path("configs/techniques/manual_strong_v1.json")
        )
        agent.reset("session", {})
        self.assertEqual(
            agent.respond("session", "I'm looking for shirts.", 1, 10)["ask_attribute"],
            "material",
        )


if __name__ == "__main__":
    unittest.main()

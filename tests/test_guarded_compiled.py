from __future__ import annotations

import gc
import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from ghostlab.policy.models import RuntimeConfig
from ghostlab.runtime.agent import GhostLabRuntime
from ghostlab.runtime.guarded_gbdt import (
    CompiledGuardedGBDTAgent,
    has_observable_override,
)
from ghostlab.state.memory import ConversationState

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = ROOT / "configs/techniques/guarded_constraint_gbdt_v1.json"


class GuardedCompiledRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = GhostLabRuntime(ROOT / "data/catalog.jsonl", CANDIDATE_CONFIG)
        primary = cls.runtime._primary
        if not isinstance(primary, CompiledGuardedGBDTAgent):
            raise TypeError("candidate did not compile to the guarded runtime")
        cls.primary = primary
        cls.models_were_lazy = primary.models._base is None

    def test_01_models_are_lazy_content_addressed_and_sequence_exact(self) -> None:
        self.assertTrue(self.models_were_lazy)
        self.runtime.reset("lazy", {})
        questions = [
            self.runtime.respond("lazy", message, turn, 10)["ask_attribute"]
            for turn, message in enumerate(
                (
                    "I'm looking for shirts, but I'm still exploring.",
                    "For that, what matters is: cotton.",
                    "I don't have an additional preference for other.",
                ),
                1,
            )
        ]
        self.assertEqual(questions, ["other", "other", "use_case"])
        self.assertIsNotNone(self.primary.models._base)
        self.assertIsNotNone(self.primary.models._constraint)
        state = self.primary._sessions["lazy"].state
        self.assertEqual(state.asked_attributes, ["other", "use_case"])
        self.assertFalse(
            any(
                hasattr(self.primary, name)
                for name in ("routing_trace", "question_trace")
            )
        )

    def test_interleaved_and_concurrent_sessions_match_sequential_results(self) -> None:
        histories = {
            "a": (
                "I'm looking for shirts, but I'm still exploring.",
                "For that, what matters is: cotton.",
                "Actually, ignore my earlier preference. What I need is: wool.",
            ),
            "b": (
                "I'm looking for shoes, but I'm still exploring.",
                "I don't have a preference for other; please use your judgment.",
                "For that, what matters is: hiking.",
            ),
        }

        def run(session_id: str, messages: tuple[str, ...]) -> list[dict]:
            self.runtime.reset(session_id, {})
            return [
                self.runtime.respond(session_id, message, turn, 10)
                for turn, message in enumerate(messages, 1)
            ]

        expected = {
            name: run(f"sequential-{name}", messages)
            for name, messages in histories.items()
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                name: executor.submit(run, f"concurrent-{name}", messages)
                for name, messages in histories.items()
            }
            observed = {
                name: future.result(timeout=30) for name, future in futures.items()
            }
        self.assertEqual(observed, expected)

    def test_offline_runtime_makes_no_network_call(self) -> None:
        self.runtime.reset("offline", {})
        with patch(
            "socket.create_connection",
            side_effect=AssertionError("network access is forbidden"),
        ):
            response = self.runtime.respond(
                "offline", "I'm looking for shirts, but I'm still exploring.", 1, 10
            )
        self.assertEqual(len(response["recommendations"]), 10)

    def test_missing_and_corrupt_models_degrade_to_contract_safe_fallback(self) -> None:
        variants = {
            "corrupt": {"sha256": "0" * 64},
            "missing": {
                "path": "artifacts/models/missing_guarded_model.json",
                "sha256": "0" * 64,
            },
        }
        for name, replacement in variants.items():
            with self.subTest(name=name):
                config = json.loads(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
                config["techniques"]["base_model_asset"].update(replacement)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
                    json.dump(config, handle)
                    handle.flush()
                    runtime = GhostLabRuntime(ROOT / "data/catalog.jsonl", handle.name)
                    runtime.reset("fallback", {})
                    response = runtime.respond(
                        "fallback",
                        "I'm looking for shirts, but I'm still exploring.",
                        1,
                        10,
                    )
                self.assertEqual(response["ask_attribute"], "material")
                self.assertEqual(len(response["recommendations"]), 10)
                del runtime
                gc.collect()

    def test_candidate_asset_hashes_match_files(self) -> None:
        config = RuntimeConfig.model_validate_json(
            CANDIDATE_CONFIG.read_text(encoding="utf-8")
        )
        for asset in (
            config.techniques.base_model_asset,
            config.techniques.constraint_model_asset,
        ):
            assert asset is not None
            self.assertEqual(
                hashlib.sha256((ROOT / asset.path).read_bytes()).hexdigest(),
                asset.sha256,
            )

    def test_guard_uses_only_the_frozen_observable_invalidation_reasons(self) -> None:
        ordinary = ConversationState("ordinary", {})
        ordinary.observe("I'm looking for shoes. What I need is: black.", 1)
        ordinary.observe("What I need is: navy.", 2)
        self.assertFalse(has_observable_override(ordinary))

        explicit = ConversationState("explicit", {})
        explicit.observe("I'm looking for shoes. What I need is: black.", 1)
        explicit.observe("Actually, what I need is: navy.", 2)
        self.assertTrue(has_observable_override(explicit))

        reset = ConversationState("reset", {})
        reset.observe("I'm looking for shoes. What I need is: black.", 1)
        reset.observe("Please start over.", 2)
        self.assertTrue(has_observable_override(reset))

        earlier = ConversationState("earlier", {})
        earlier.observe("I'm looking for shoes. What I need is: black.", 1)
        earlier.observe(
            "Actually, ignore my earlier preference. What I need is: navy.", 2
        )
        self.assertTrue(has_observable_override(earlier))

        category = ConversationState("category", {})
        category.observe("I'm looking for shoes, but I'm still exploring.", 1)
        category.observe("Actually, I'm looking for shirts instead.", 2)
        self.assertTrue(has_observable_override(category))


if __name__ == "__main__":
    unittest.main()

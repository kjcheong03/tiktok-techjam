from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ghostlab.retrieval.constraint_gbdt import (
    ConstraintAgentAdapter,
    ConstraintContext,
    ConstraintGBDTFeatureStore,
    RuntimeConstraintReranker,
)
from ghostlab.retrieval.gbdt import LambdaMARTModel
from ghostlab.runtime.experimental import ExperimentalAgent
from ghostlab.state.memory import ConversationState
from scripts.run_gbdt_constraint_interaction import (
    FIELD_WEIGHTS,
    QUESTION_ORDER,
    record_training_question,
)


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

    def test_training_question_bookkeeping_matches_runtime_exactly(self) -> None:
        messages = (
            "I'm looking for a coat, but I'm still exploring.",
            "I don't have a preference for other; please use your judgment.",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog(directory)
            agent = ExperimentalAgent(
                path,
                state_variant="raw_history",
                question_variant="sequence",
                question_order=QUESTION_ORDER,
            )
            agent.reset("runtime", {})
            runtime = agent.sessions["runtime"]
            self.assertIsInstance(runtime, ConversationState)
            training = ConversationState("training", {}, multi_value=False)
            for turn, message in enumerate(messages, 1):
                assert isinstance(runtime, ConversationState)
                agent._query_and_question(runtime, message, turn)
                training.observe(message, turn)
                question = QUESTION_ORDER[turn - 1]
                record_training_question(training, question)
            assert isinstance(runtime, ConversationState)
            runtime_context = ConstraintContext.from_runtime(
                runtime, turn=2, retrieval_scores=[4.0, 2.0]
            )
            training_context = ConstraintContext.from_runtime(
                training, turn=2, retrieval_scores=[4.0, 2.0]
            )
        self.assertEqual(runtime_context, training_context)
        self.assertEqual(runtime_context.asked_count, 1)
        self.assertEqual(runtime_context.no_preference_count, 1)

    def test_runtime_context_is_isolated_across_concurrent_sessions(self) -> None:
        barrier = threading.Barrier(2)
        calls: list[tuple[str, int]] = []
        lock = threading.Lock()

        class CapturingReranker:
            def rerank_with_context(
                self,
                query: str,
                ranking: list[str],
                *,
                state: ConversationState,
                turn: int,
                retrieval_scores: list[float],
                rerank_k: int = 50,
            ) -> list[str]:
                del query, retrieval_scores, rerank_k
                barrier.wait(timeout=5)
                with lock:
                    calls.append((state.session_id, turn))
                return ranking

        class FakeAgent:
            def __init__(
                self,
                runtime: RuntimeConstraintReranker,
                states: dict[str, ConversationState],
            ) -> None:
                self.runtime = runtime
                self.sessions = states

            def reset(self, session_id: str, user_profile: dict) -> None:
                self.sessions[session_id] = ConversationState(session_id, user_profile)

            def respond(
                self, session_id: str, user_message: str, turn: int, top_k: int
            ) -> dict:
                del session_id, user_message, turn, top_k
                ranking = self.runtime.rerank("coat", ["wool", "cotton"])
                return {"message": "ok", "recommendations": ranking}

        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog(directory)
            store = ConstraintGBDTFeatureStore(path)
            model = LambdaMARTModel(
                candidate_id="context-test",
                feature_names=("original_rank",),
                trees=(),
                learning_rate=0.03,
                best_iteration=0,
                training_groups=0,
                training_rows=0,
                seed=20260826,
            )
            runtime = RuntimeConstraintReranker(str(path), FIELD_WEIGHTS, store, model)
            runtime.reranker = CapturingReranker()  # type: ignore[assignment]
            states = {
                "first": ConversationState("first", {}),
                "second": ConversationState("second", {}),
            }
            adapter = ConstraintAgentAdapter(
                FakeAgent(runtime, states),  # type: ignore[arg-type]
                runtime,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(adapter.respond, "first", "a", 1, 10)
                second = executor.submit(adapter.respond, "second", "b", 7, 10)
                first.result(timeout=10)
                second.result(timeout=10)
            with self.assertRaisesRegex(RuntimeError, "was not bound"):
                runtime.rerank("coat", ["wool", "cotton"])
        self.assertCountEqual(calls, [("first", 1), ("second", 7)])

    def test_override_guard_ignores_ordinary_replacement_and_routes_override(
        self,
    ) -> None:
        class FakeRoute:
            def __init__(self, label: str) -> None:
                self.label = label
                self.calls = 0

            def rerank(
                self, query: str, ranking: list[str], *, rerank_k: int = 50
            ) -> list[str]:
                del query, rerank_k
                self.calls += 1
                return ranking

            def rerank_with_context(
                self,
                query: str,
                ranking: list[str],
                *,
                state: ConversationState,
                turn: int,
                retrieval_scores: list[float],
                rerank_k: int = 50,
            ) -> list[str]:
                del query, state, turn, retrieval_scores, rerank_k
                self.calls += 1
                return ranking

        state = ConversationState("guard", {})
        state.observe("I'm looking for shoes. What I need is: black.", 1)
        state.observe("What I need is: navy.", 2)
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog(directory)
            store = ConstraintGBDTFeatureStore(path)
            model = LambdaMARTModel(
                candidate_id="guard-test",
                feature_names=("original_rank",),
                trees=(),
                learning_rate=0.03,
                best_iteration=0,
                training_groups=0,
                training_rows=0,
                seed=20260826,
            )
            fallback = FakeRoute("base")
            runtime = RuntimeConstraintReranker(
                str(path),
                FIELD_WEIGHTS,
                store,
                model,
                fallback=fallback,  # type: ignore[arg-type]
            )
            candidate = FakeRoute("constraint")
            runtime.reranker = candidate  # type: ignore[assignment]
            with runtime.invocation(state, 2):
                runtime.rerank("shoes", ["wool", "cotton"])
            self.assertEqual(candidate.calls, 1)
            self.assertEqual(fallback.calls, 0)
            state.observe("Actually, what I need is: red.", 3)
            with runtime.invocation(state, 3):
                runtime.rerank("shoes", ["wool", "cotton"])
        self.assertEqual(candidate.calls, 1)
        self.assertEqual(fallback.calls, 1)
        self.assertEqual(
            [item["route"] for item in runtime.routing_trace],
            ["constraint", "base_override_fallback"],
        )


if __name__ == "__main__":
    unittest.main()

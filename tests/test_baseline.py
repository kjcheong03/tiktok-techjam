from __future__ import annotations

import unittest

from baseline.agent import BaselineAgent
from baseline.question_policy import fixed_other
from baseline.retrieval import reciprocal_rank_fusion
from baseline.state import SessionState
from baseline.state_v2 import StructuredSessionState


class FakeKeywordRetriever:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def search(self, session_id: str, query: str, turn: int, limit: int) -> list[str]:
        return ["A", "B", "C"]


class FakeDenseRetriever:
    def search(self, query: str, limit: int) -> list[str]:
        return ["B", "D", "A"]


class SequenceKeywordRetriever:
    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def search(self, session_id: str, query: str, turn: int, limit: int) -> list[str]:
        return list(self.ranking)


class StateTest(unittest.TestCase):
    def test_accumulates_category_and_constraint(self) -> None:
        state = SessionState("s", {})
        state.observe(
            "I'm looking for running shoes. A key requirement is: black leather.",
            1,
        )
        query = state.build_query().lower()
        self.assertIn("running shoes", query)
        self.assertIn("black leather", query)

    def test_override_invalidates_stale_preference_but_keeps_category(self) -> None:
        state = SessionState("s", {})
        state.observe("I'm looking for shoes. color: black", 1)
        state.observe(
            "Actually, ignore my earlier preference. What I need is: color: navy.",
            3,
        )
        query = state.build_query().lower()
        self.assertIn("shoes", query)
        self.assertIn("navy", query)
        self.assertNotIn("black", query)
        active = [slot for slot in state.slots if slot.active]
        self.assertEqual(active[-1].attribute, "color")

    def test_no_preference_prevents_repeated_question(self) -> None:
        state = SessionState("s", {})
        self.assertEqual(state.choose_question(), "material")
        state.observe("I don't have a preference for material; please use your judgment.", 2)
        self.assertEqual(state.choose_question(), "color")


class RetrievalTest(unittest.TestCase):
    def test_agent_accepts_an_injected_question_policy(self) -> None:
        keyword = FakeKeywordRetriever()
        agent = BaselineAgent(
            mode="keyword",
            stateful=True,
            keyword=keyword,  # type: ignore[arg-type]
            dense=None,
            question_policy=fixed_other,
        )
        agent.reset("session", {})

        response = agent.respond("session", "I'm looking for shoes.", 1, 10)

        self.assertEqual(response["ask_attribute"], "other")
        self.assertEqual(agent.sessions["session"].last_asked_attribute, "other")

    def test_rrf_rewards_agreement_and_is_deterministic(self) -> None:
        result = reciprocal_rank_fusion(
            [["A", "B", "C"], ["B", "D", "A"]], rank_constant=60, limit=4
        )
        self.assertEqual(result, ["B", "A", "D", "C"])

    def test_hybrid_agent_returns_contract_shape(self) -> None:
        keyword = FakeKeywordRetriever()
        agent = BaselineAgent(
            mode="hybrid",
            stateful=True,
            keyword=keyword,  # type: ignore[arg-type]
            dense=FakeDenseRetriever(),  # type: ignore[arg-type]
        )
        agent.reset("session", {})
        response = agent.respond("session", "I'm looking for shoes, but I'm still exploring.", 1, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "B")
        self.assertIn(response["ask_attribute"], {"material", "color"})
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_opt_out_repeats_ranked_candidates(self) -> None:
        agent = BaselineAgent(
            mode="keyword",
            stateful=True,
            keyword=SequenceKeywordRetriever(["A", "B", "C"]),  # type: ignore[arg-type]
            dense=None,
            state_factory=StructuredSessionState,
        )
        agent.reset("session", {})

        first = agent.respond("session", "I'm looking for shoes.", 1, 2)
        second = agent.respond("session", "I'm still exploring.", 2, 2)

        self.assertEqual(first["recommendations"], [{"parent_asin": "A"}, {"parent_asin": "B"}])
        self.assertEqual(second["recommendations"], [{"parent_asin": "A"}, {"parent_asin": "B"}])

    def test_opt_in_filters_seen_and_fills_from_retrieval_window(self) -> None:
        agent = BaselineAgent(
            mode="keyword",
            stateful=True,
            keyword=SequenceKeywordRetriever(["A", "B", "C", "D"]),  # type: ignore[arg-type]
            dense=None,
            state_factory=StructuredSessionState,
            filter_seen_recommendations=True,
        )
        agent.reset("session", {})

        first = agent.respond("session", "I'm looking for shoes.", 1, 2)
        second = agent.respond("session", "I'm still exploring.", 2, 2)

        self.assertEqual(first["recommendations"], [{"parent_asin": "A"}, {"parent_asin": "B"}])
        self.assertEqual(second["recommendations"], [{"parent_asin": "C"}, {"parent_asin": "D"}])

    def test_only_returned_recommendations_are_recorded(self) -> None:
        agent = BaselineAgent(
            mode="keyword",
            stateful=True,
            keyword=SequenceKeywordRetriever(["A", "B", "C"]),  # type: ignore[arg-type]
            dense=None,
            state_factory=StructuredSessionState,
            filter_seen_recommendations=True,
        )
        agent.reset("session", {})

        agent.respond("session", "I'm looking for shoes.", 1, 2)

        state = agent.sessions["session"]
        self.assertEqual(state.shown_product_ids, {"A", "B"})
        self.assertNotIn("C", state.shown_product_ids)

    def test_opt_in_filtering_applies_to_dense_and_hybrid_rankings(self) -> None:
        for mode, keyword, dense, expected_first, expected_second in (
            ("dense", SequenceKeywordRetriever([]), FakeDenseRetriever(), ["B", "D"], ["A"]),
            ("hybrid", SequenceKeywordRetriever(["A", "B", "C"]), FakeDenseRetriever(), ["B", "A"], ["D", "C"]),
        ):
            with self.subTest(mode=mode):
                agent = BaselineAgent(
                    mode=mode,
                    stateful=True,
                    keyword=keyword,  # type: ignore[arg-type]
                    dense=dense,  # type: ignore[arg-type]
                    retrieval_k=4,
                    state_factory=StructuredSessionState,
                    filter_seen_recommendations=True,
                )
                agent.reset("session", {})

                first = agent.respond("session", "I'm looking for shoes.", 1, 2)
                second = agent.respond("session", "I'm still exploring.", 2, 2)

                self.assertEqual(
                    [item["parent_asin"] for item in first["recommendations"]],
                    expected_first,
                )
                self.assertEqual(
                    [item["parent_asin"] for item in second["recommendations"]],
                    expected_second,
                )


if __name__ == "__main__":
    unittest.main()

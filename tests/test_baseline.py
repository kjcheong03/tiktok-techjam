from __future__ import annotations

import unittest

from baseline.agent import BaselineAgent
from baseline.retrieval import reciprocal_rank_fusion
from baseline.state import SessionState


class FakeKeywordRetriever:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def search(self, session_id: str, query: str, turn: int, limit: int) -> list[str]:
        return ["A", "B", "C"]


class FakeDenseRetriever:
    def search(self, query: str, limit: int) -> list[str]:
        return ["B", "D", "A"]


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


if __name__ == "__main__":
    unittest.main()

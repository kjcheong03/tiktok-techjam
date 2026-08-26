from __future__ import annotations

import unittest

from baseline.question_policy import current_order, fixed_other
from baseline.raw_history_control import RawHistoryNoManagedStateAgent


class FakeKeywordRetriever:
    def __init__(self) -> None:
        self.reset_calls: list[tuple[str, dict]] = []
        self.search_calls: list[tuple[str, str, int, int]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.reset_calls.append((session_id, user_profile))

    def search(self, session_id: str, query: str, turn: int, limit: int) -> list[str]:
        self.search_calls.append((session_id, query, turn, limit))
        return ["A", "B", "C"]


class RawHistoryNoManagedStateAgentTest(unittest.TestCase):
    def test_search_uses_exact_accumulated_raw_query(self) -> None:
        keyword = FakeKeywordRetriever()
        agent = RawHistoryNoManagedStateAgent(keyword, current_order)  # type: ignore[arg-type]
        agent.reset("session", {})

        agent.respond("session", "I need trail shoes.", 1, 10)
        agent.respond("session", "Must be black", 2, 10)

        self.assertEqual(
            [call[1] for call in keyword.search_calls],
            ["I need trail shoes.", "I need trail shoes.. Must be black"],
        )
        self.assertTrue(all(call[3] == 200 for call in keyword.search_calls))

    def test_current_order_uses_original_fixed_turn_questions(self) -> None:
        keyword = FakeKeywordRetriever()
        agent = RawHistoryNoManagedStateAgent(keyword, current_order)  # type: ignore[arg-type]
        agent.reset("session", {})

        questions = [
            agent.respond("session", f"message {turn}", turn, 10)["ask_attribute"]
            for turn in range(1, 9)
        ]

        self.assertEqual(
            questions,
            ["material", "color", "style", "use_case", "feature", "budget", "size", None],
        )

    def test_fixed_other_uses_injected_policy_on_every_turn(self) -> None:
        keyword = FakeKeywordRetriever()
        agent = RawHistoryNoManagedStateAgent(keyword, fixed_other)  # type: ignore[arg-type]
        agent.reset("session", {})

        questions = [
            agent.respond("session", f"message {turn}", turn, 10)["ask_attribute"]
            for turn in (1, 2, 10)
        ]

        self.assertEqual(questions, ["other", "other", "other"])

    def test_reset_isolates_sessions_and_clears_prior_history(self) -> None:
        keyword = FakeKeywordRetriever()
        agent = RawHistoryNoManagedStateAgent(keyword, current_order)  # type: ignore[arg-type]
        agent.reset("first", {})
        agent.reset("second", {})

        agent.respond("first", "first message", 1, 10)
        agent.respond("second", "second message", 1, 10)
        agent.reset("first", {})
        agent.respond("first", "replacement", 1, 10)

        self.assertEqual(
            [call[:2] for call in keyword.search_calls],
            [
                ("first", "first message"),
                ("second", "second message"),
                ("first", "replacement"),
            ],
        )

    def test_response_matches_contract_with_zero_usage(self) -> None:
        keyword = FakeKeywordRetriever()
        agent = RawHistoryNoManagedStateAgent(keyword, current_order)  # type: ignore[arg-type]
        agent.reset("session", {})

        response = agent.respond("session", "hello", 1, 2)

        self.assertEqual(
            response,
            {
                "message": "Do you have a preference for material?",
                "ask_attribute": "material",
                "recommendations": [{"parent_asin": "A"}, {"parent_asin": "B"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            },
        )


if __name__ == "__main__":
    unittest.main()

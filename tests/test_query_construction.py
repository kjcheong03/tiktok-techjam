from __future__ import annotations

import unittest

from ghostlab.state.memory import ConversationState
from ghostlab.state.query import build_query


class QueryConstructionTest(unittest.TestCase):
    def test_raw_history_preserves_turns(self) -> None:
        state = ConversationState("s1", {})
        state.observe("I'm looking for running shoes.", 1)
        state.observe("For that, what matters is: wide fit.", 2)
        self.assertEqual(
            build_query(state, "raw_history"),
            "I'm looking for running shoes. For that, what matters is: wide fit",
        )

    def test_negation_safe_hybrid_uses_active_override(self) -> None:
        state = ConversationState("s1", {})
        state.observe("I'm looking for boots. black", 1)
        state.observe(
            "Actually, ignore my earlier preference. What I need is: brown leather.",
            2,
        )
        query = build_query(state, "negation_safe_hybrid").casefold()
        self.assertIn("brown leather", query)
        self.assertNotIn("black", query)

    def test_compressed_raw_keeps_first_and_latest_turns(self) -> None:
        state = ConversationState("s1", {})
        for turn in range(1, 7):
            state.observe(f"message {turn}", turn)
        query = build_query(state, "compressed_raw")
        self.assertIn("message 1", query)
        self.assertNotIn("message 2", query)
        self.assertIn("message 6", query)


if __name__ == "__main__":
    unittest.main()

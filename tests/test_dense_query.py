from __future__ import annotations

import unittest

from ghostlab.retrieval.query import active_structured_query, build_dense_query
from ghostlab.state.memory import ConversationState


class DenseQueryTest(unittest.TestCase):
    def test_raw_plus_active_is_non_destructive(self) -> None:
        state = ConversationState("s", {})
        state.observe(
            "I'm looking for trail shoes. A key requirement is: waterproof.", 1
        )
        query = build_dense_query(state, "raw_plus_active")
        self.assertIn("I'm looking for trail shoes", query)
        self.assertIn("Active preferences:", query)
        self.assertIn("waterproof", query)

    def test_structured_query_excludes_negative_and_no_preference_prose(self) -> None:
        state = ConversationState("s", {})
        state.observe(
            "I'm looking for trail shoes. A key requirement is: waterproof.", 1
        )
        state.observe("Actually, avoid waterproof. I want breathable mesh instead.", 2)
        state.no_preference_attributes.add("material")
        query = build_dense_query(state, "negation_safe_structured").casefold()
        self.assertNotIn("avoid", query)
        self.assertNotIn("no preference", query)
        self.assertNotIn("waterproof", query)

    def test_structured_query_has_labeled_active_values(self) -> None:
        state = ConversationState("s", {})
        state.observe("I'm looking for running shoes, but I'm still exploring.", 1)
        self.assertIn("Category:", active_structured_query(state))

    def test_empty_structure_falls_back_to_current_message(self) -> None:
        state = ConversationState("s", {})
        state.observe("Show me something interesting", 1)
        self.assertEqual(
            build_dense_query(state, "negation_safe_structured"),
            "Show me something interesting",
        )


if __name__ == "__main__":
    unittest.main()

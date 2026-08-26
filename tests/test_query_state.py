from __future__ import annotations

import unittest

from baseline.query_state import RawHistorySessionState


class RawHistorySessionStateTest(unittest.TestCase):
    def test_query_uses_exact_multiple_message_history(self) -> None:
        state = RawHistorySessionState("session", {})
        messages = ["I need trail shoes.", "Must be black", "Actually, navy please."]

        for turn, message in enumerate(messages, 1):
            state.observe(message, turn, parsed_constraints=[])

        self.assertEqual(
            state.build_query(),
            "I need trail shoes.. Must be black. Actually, navy please.",
        )

    def test_inherited_constraints_and_targeted_correction_still_update(self) -> None:
        state = RawHistorySessionState("session", {})
        state.observe("I'm looking for shoes. black.", 1)
        state.observe("A key requirement is: leather; under $80.", 1)
        state.observe(
            "Actually, ignore my earlier preference. What I need is: navy.",
            2,
        )

        color = [item for item in state.constraints if item.attribute == "color"]
        self.assertEqual([item.values for item in color], [["black"], ["navy"]])
        self.assertEqual(color[0].status, "superseded")
        self.assertEqual(color[1].status, "active")
        self.assertTrue(any(item.attribute == "material" and item.active for item in state.constraints))
        self.assertTrue(any(item.attribute == "budget" and item.active for item in state.constraints))

    def test_reset_clears_raw_history(self) -> None:
        state = RawHistorySessionState("session", {})
        state.observe("I need trail shoes.", 1, parsed_constraints=[])

        state.reset("next", {"summary": "new profile"})

        self.assertEqual(state.messages, [])
        self.assertEqual(state.build_query(), "")


if __name__ == "__main__":
    unittest.main()

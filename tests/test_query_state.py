from __future__ import annotations

import unittest

from baseline.constraints import StructuredConstraint
from baseline.query_state import (
    RawHistorySessionState,
    StatePrioritizedRawHistorySessionState,
)


def evidence(
    attribute: str,
    values: list[str],
    turn: int,
    text: str,
    **kwargs: str,
) -> StructuredConstraint:
    return StructuredConstraint(
        attribute=attribute,  # type: ignore[arg-type]
        values=values,
        source_turn=turn,
        source_text=text,
        **kwargs,
    )


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


class StatePrioritizedRawHistorySessionStateTest(unittest.TestCase):
    def test_active_state_prefix_and_exact_raw_suffix(self) -> None:
        state = StatePrioritizedRawHistorySessionState("session", {})
        messages = ["I need shoes.", "Something durable, please."]
        for turn, message in enumerate(messages, 1):
            state.observe(message, turn, parsed_constraints=[])
        state.apply_constraints(
            [
                evidence("category", ["shoes"], 1, messages[0]),
                evidence("material", ["leather"], 2, "leather"),
            ]
        )

        raw_history = ". ".join(messages)
        self.assertEqual(
            state.build_query(),
            f"shoes. leather. {raw_history}",
        )

    def test_correction_prioritizes_replacement_but_keeps_old_raw_value(self) -> None:
        state = StatePrioritizedRawHistorySessionState("session", {})
        messages = [
            "I'm looking for shoes. black.",
            "Actually, ignore my earlier preference. What I need is: navy.",
        ]
        for turn, message in enumerate(messages, 1):
            state.observe(message, turn)

        raw_history = ". ".join(messages)
        self.assertEqual(
            state.build_query(),
            f"shoes. navy. {raw_history}",
        )

    def test_no_preference_preserves_query_evidence_and_raw_transcript(self) -> None:
        state = StatePrioritizedRawHistorySessionState("session", {})
        state.apply_constraints(
            [evidence("material", ["leather"], 1, "I need leather.")]
        )
        no_preference = "I don't have a preference for material; please use your judgment."
        state.observe(no_preference, 2)

        self.assertIn("leather", state.build_query())
        self.assertIn(no_preference, state.build_query())
        self.assertEqual(state.choose_question(), "color")

    def test_empty_active_state_falls_back_to_raw_history(self) -> None:
        state = StatePrioritizedRawHistorySessionState("session", {})
        state.observe("I need trail shoes.", 1, parsed_constraints=[])

        self.assertEqual(state.build_query(), "I need trail shoes.")

    def test_reset_clears_state_and_raw_history(self) -> None:
        state = StatePrioritizedRawHistorySessionState("session", {})
        state.observe("I need shoes.", 1, parsed_constraints=[])
        state.apply_constraints([evidence("material", ["leather"], 1, "leather")])

        state.reset("next", {"summary": "new profile"})

        self.assertEqual(state.messages, [])
        self.assertEqual(state.constraints, [])
        self.assertEqual(state.build_query(), "")


if __name__ == "__main__":
    unittest.main()

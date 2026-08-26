from __future__ import annotations

import unittest

from baseline.constraints import StructuredConstraint
from baseline.query_state import (
    RawHistorySessionState,
    StateConsumedRawHistorySessionState,
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


class StateConsumedRawHistorySessionStateTest(unittest.TestCase):
    def test_query_preserves_raw_lexical_evidence_in_source_order(self) -> None:
        state = StateConsumedRawHistorySessionState("session", {})
        state.observe("Need Trail-shoes!", 1, parsed_constraints=[])
        state.observe("I prefer blue, lightweight.", 2, parsed_constraints=[])

        self.assertEqual(
            state.build_query(),
            "need trail shoes i prefer blue lightweight",
        )

    def test_superseded_include_is_removed_and_active_value_is_kept(self) -> None:
        state = StateConsumedRawHistorySessionState("session", {})
        state.observe(
            "I want black shoes.",
            1,
            parsed_constraints=[evidence("color", ["black"], 1, "I want black shoes.")],
        )
        state.observe(
            "Actually, navy shoes.",
            2,
            parsed_constraints=[
                evidence("color", ["navy"], 2, "Actually, navy shoes."),
            ],
        )

        self.assertEqual(state.build_query(), "i want shoes actually navy shoes")

    def test_legacy_adapter_correction_changes_the_raw_history_query(self) -> None:
        state = StateConsumedRawHistorySessionState("session", {})
        state.observe("I'm looking for shoes. black.", 1)
        state.observe(
            "Actually, ignore my earlier preference. What I need is: navy.",
            3,
        )

        terms = state.build_query().split()
        self.assertIn("shoes", terms)
        self.assertIn("navy", terms)
        self.assertNotIn("black", terms)

    def test_active_negative_terms_are_removed(self) -> None:
        state = StateConsumedRawHistorySessionState("session", {})
        state.observe(
            "I need leather shoes, but not red.",
            1,
            parsed_constraints=[
                evidence("material", ["leather"], 1, "I need leather shoes, but not red."),
                evidence(
                    "color",
                    ["red"],
                    1,
                    "I need leather shoes, but not red.",
                    polarity="exclude",
                ),
            ],
        )

        self.assertEqual(state.build_query(), "i need leather shoes but not")

    def test_no_preference_removes_value_and_attribute_name_tokens(self) -> None:
        state = StateConsumedRawHistorySessionState("session", {})
        state.observe(
            "I need leather shoes.",
            1,
            parsed_constraints=[evidence("material", ["leather"], 1, "leather")],
        )
        state.observe(
            "Material is not important.",
            2,
            parsed_constraints=[],
            no_preference_attributes=["material"],
        )

        self.assertEqual(state.build_query(), "i need shoes is not important")

    def test_shared_token_is_preserved_when_also_active_positive(self) -> None:
        state = StateConsumedRawHistorySessionState("session", {})
        state.observe(
            "Black shoes.",
            1,
            parsed_constraints=[evidence("color", ["black"], 1, "Black shoes.")],
        )
        state.supersede_attribute("color")
        state.observe(
            "A black style.",
            2,
            parsed_constraints=[evidence("style", ["black"], 2, "A black style.")],
        )

        self.assertEqual(state.build_query().split().count("black"), 2)

    def test_active_state_only_term_is_appended(self) -> None:
        state = StateConsumedRawHistorySessionState("session", {})
        state.observe(
            "I need shoes.",
            1,
            parsed_constraints=[
                evidence("category", ["shoes"], 1, "I need shoes."),
                evidence("material", ["leather"], 1, "I need shoes."),
            ],
        )

        self.assertEqual(state.build_query(), "i need shoes leather")

    def test_reset_clears_raw_history_and_constraints(self) -> None:
        state = StateConsumedRawHistorySessionState("session", {})
        state.observe(
            "I need leather shoes.",
            1,
            parsed_constraints=[evidence("material", ["leather"], 1, "leather")],
        )

        state.reset("next", {"summary": "new profile"})

        self.assertEqual(state.messages, [])
        self.assertEqual(state.constraints, [])
        self.assertEqual(state.build_query(), "")


if __name__ == "__main__":
    unittest.main()

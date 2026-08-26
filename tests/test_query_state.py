from __future__ import annotations

import unittest

from baseline.constraints import StructuredConstraint
from baseline.query_state import (
    CoverageAdaptiveSessionState,
    RawHistorySessionState,
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


class CoverageAdaptiveSessionStateTest(unittest.TestCase):
    def test_normal_multi_value_state_uses_clean_state_only_query(self) -> None:
        state = CoverageAdaptiveSessionState("session", {})
        state.observe(
            "I'm looking for shirts. A key requirement is: black; navy.",
            1,
        )

        self.assertEqual(state.build_query(), "shirts. black. navy")

    def test_low_coverage_correction_returns_exact_raw_history(self) -> None:
        state = CoverageAdaptiveSessionState("session", {})
        messages = [
            "I'm looking for shoes. black.",
            "Actually, ignore my earlier preference. What I need is: navy.",
        ]
        for turn, message in enumerate(messages, 1):
            state.observe(message, turn)

        self.assertEqual(state.build_query(), ". ".join(messages))

    def test_correction_with_four_active_constraints_stays_state_only(self) -> None:
        state = CoverageAdaptiveSessionState("session", {})
        state.apply_constraints(
            [
                evidence("category", ["shoes"], 1, "shoes"),
                evidence("material", ["leather"], 1, "leather"),
                evidence("color", ["black"], 1, "black"),
                evidence("style", ["casual"], 1, "casual"),
            ]
        )
        state.observe(
            "Actually, navy.",
            2,
            parsed_constraints=[evidence("color", ["navy"], 2, "Actually, navy.")],
        )

        self.assertEqual(
            state.build_query(),
            "shoes. leather. casual. navy",
        )

    def test_ambiguous_correction_with_no_superseded_evidence_stays_state_only(self) -> None:
        state = CoverageAdaptiveSessionState("session", {})
        state.observe("I'm looking for shoes. A key requirement is: leather.", 1)
        state.observe(
            "Actually, ignore my earlier preference. What I need is: something.",
            2,
        )

        self.assertEqual(state.build_query(), "shoes. leather")

    def test_empty_active_state_falls_back_to_raw_history(self) -> None:
        state = CoverageAdaptiveSessionState("session", {})
        state.observe("I need trail shoes.", 1, parsed_constraints=[])

        self.assertEqual(state.build_query(), "I need trail shoes.")

    def test_reset_clears_state_and_raw_history(self) -> None:
        state = CoverageAdaptiveSessionState("session", {})
        state.observe("I need shoes.", 1, parsed_constraints=[])
        state.apply_constraints([evidence("material", ["leather"], 1, "leather")])

        state.reset("next", {"summary": "new profile"})

        self.assertEqual(state.messages, [])
        self.assertEqual(state.constraints, [])
        self.assertEqual(state.build_query(), "")


if __name__ == "__main__":
    unittest.main()

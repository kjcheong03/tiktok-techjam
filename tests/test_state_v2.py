from __future__ import annotations

import unittest

from baseline.constraints import LegacyV1ConstraintAdapter, StructuredConstraint
from baseline.state_v2 import StructuredSessionState


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


class ConstraintAdapterTest(unittest.TestCase):
    def test_constraint_contract_retains_structured_fields(self) -> None:
        constraint = StructuredConstraint(
            attribute="budget",
            values=[80],  # type: ignore[list-item]
            relation="any",
            polarity="include",
            strength="hard",
            operator="at_most",
            source_turn=4,
            source_text="It must be under $80",
            provenance="explicit",
        )

        self.assertEqual(constraint.values, ["80"])
        self.assertEqual(constraint.operator, "at_most")
        self.assertEqual(constraint.strength, "hard")
        self.assertEqual(constraint.status, "active")

    def test_adapter_delegates_to_v1_templates_and_preserves_source(self) -> None:
        message = "I'm looking for trail shoes. A key requirement is: black leather."
        constraints = LegacyV1ConstraintAdapter().parse(message, 1)

        self.assertEqual([item.attribute for item in constraints], ["category", "material"])
        self.assertEqual(constraints[0].values, ["trail shoes"])
        self.assertEqual(constraints[1].values, ["black leather"])
        self.assertEqual(constraints[1].source_text, message)
        self.assertEqual(constraints[1].provenance, "explicit")

    def test_adapter_uses_last_asked_attribute_for_v1_answer(self) -> None:
        message = "For that, what matters is: blue."
        constraints = LegacyV1ConstraintAdapter().parse(
            message,
            2,
            last_asked_attribute="color",
        )

        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].attribute, "color")
        self.assertEqual(constraints[0].provenance, "simulator_answer")

    def test_adapter_keeps_all_same_attribute_feature_answer_values_active(self) -> None:
        message = "For that, what matters is: Imported; Wrap closure."
        constraints = LegacyV1ConstraintAdapter().parse(
            message,
            2,
            last_asked_attribute="feature",
        )

        self.assertEqual([item.attribute for item in constraints], ["feature", "feature"])
        self.assertEqual(
            [item.values for item in constraints],
            [["imported"], ["wrap closure"]],
        )
        self.assertTrue(all(item.active for item in constraints))
        self.assertTrue(all(item.provenance == "simulator_answer" for item in constraints))

    def test_adapter_keeps_all_same_attribute_other_answer_values_active(self) -> None:
        message = "For that, what matters is: 96% Nylon, 4% Spandex; Pull-On closure"
        constraints = LegacyV1ConstraintAdapter().parse(
            message,
            2,
            last_asked_attribute="other",
        )

        self.assertEqual([item.attribute for item in constraints], ["other", "other"])
        self.assertEqual(
            [item.values for item in constraints],
            [["96% nylon, 4% spandex"], ["pull-on closure"]],
        )
        self.assertTrue(all(item.active for item in constraints))
        self.assertTrue(all(item.provenance == "simulator_answer" for item in constraints))


class StateV2ReplayTest(unittest.TestCase):
    def test_multi_value_other_answer_survives_no_additional_preference(self) -> None:
        state = StructuredSessionState("session", {})
        state.last_asked_attribute = "other"
        state.observe(
            "For that, what matters is: 96% Nylon, 4% Spandex; Pull-On closure.",
            2,
        )
        state.last_asked_attribute = "other"
        state.observe(
            "I don't have an additional preference for other.",
            3,
        )

        self.assertEqual(
            state.active_values("other"),
            ["96% nylon, 4% spandex", "pull-on closure"],
        )
        query = state.build_query()
        self.assertIn("96% nylon, 4% spandex", query)
        self.assertIn("pull-on closure", query)

    def test_replay_accumulates_compatible_values_and_deduplicates(self) -> None:
        state = StructuredSessionState("session", {})
        state.observe(
            "I'm looking for shirts.",
            1,
            parsed_constraints=[evidence("category", ["shirts"], 1, "I'm looking for shirts.")],
        )
        state.observe(
            "For that, what matters is: black; navy.",
            2,
            parsed_constraints=[
                evidence("color", ["black"], 2, "For that, what matters is: black; navy."),
                evidence("color", ["navy"], 2, "For that, what matters is: black; navy."),
            ],
        )
        state.observe(
            "For that, what matters is: navy.",
            3,
            parsed_constraints=[evidence("color", ["NAVY"], 3, "For that, what matters is: navy.")],
        )

        self.assertEqual(state.active_values("color"), ["black", "navy"])
        self.assertEqual(state.build_query(), "shirts. black. navy")
        self.assertEqual(len(state.active_constraints), 3)

    def test_targeted_correction_preserves_unrelated_constraints(self) -> None:
        state = StructuredSessionState("session", {})
        state.observe(
            "I'm looking for shoes. black.",
            1,
        )
        state.observe("A key requirement is: leather; under $80.", 1)
        state.record_recommendations(["shown-before-correction"])
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
        self.assertIn("shoes", state.build_query())
        self.assertNotIn("black", state.build_query())
        self.assertIn("navy", state.build_query())
        self.assertIn("I'm looking for shoes", state.messages[0])
        self.assertEqual(state.shown_product_ids, set())

    def test_ambiguous_correction_preserves_state_and_raw_message(self) -> None:
        state = StructuredSessionState("session", {})
        state.observe(
            "I'm looking for shoes. A key requirement is: black leather.",
            1,
        )
        state.record_recommendations(["shown-before-ambiguous-correction"])
        ambiguous = "Actually, ignore my earlier preference. What I need is: something."
        state.observe(ambiguous, 2)

        self.assertEqual(state.build_query(), "shoes. black leather")
        self.assertTrue(all(item.status == "active" for item in state.constraints))
        self.assertEqual(state.messages[-1], ambiguous)
        self.assertEqual(state.shown_product_ids, {"shown-before-ambiguous-correction"})

    def test_no_preference_skips_question_but_keeps_query_value(self) -> None:
        state = StructuredSessionState("session", {})
        state.apply_constraints(
            [evidence("material", ["leather"], 1, "I need leather.", strength="hard")]
        )
        state.observe(
            "I don't have a preference for material; please use your judgment.",
            2,
        )

        self.assertIn("material", state.no_preference_attributes)
        self.assertTrue(state.constraints[0].active)
        self.assertIn("leather", state.build_query())
        self.assertEqual(state.choose_question(), "color")

        state.apply_constraints(
            [evidence("material", ["cotton"], 3, "I prefer cotton.")]
        )
        self.assertNotIn("material", state.no_preference_attributes)
        self.assertIn("leather", state.build_query())
        self.assertIn("cotton", state.build_query())

    def test_query_order_and_omissions_are_deterministic(self) -> None:
        state = StructuredSessionState("session", {})
        state.apply_constraints(
            [
                evidence("color", ["navy", "NAVY"], 4, "navy"),
                evidence("category", ["jackets"], 5, "jackets"),
                evidence("material", ["cotton"], 2, "cotton"),
                evidence("material", ["leather"], 3, "not leather", polarity="exclude"),
                evidence("style", ["old"], 1, "old", status="superseded"),
            ]
        )

        self.assertEqual(state.build_query(), "jackets. cotton. navy")

    def test_reset_clears_conversation_state(self) -> None:
        state = StructuredSessionState("session", {})
        state.observe("I'm looking for shoes.", 1)
        state.record_recommendations(["A", "B"])

        state.reset("next", {"summary": "new profile"})

        self.assertEqual(state.session_id, "next")
        self.assertEqual(state.user_profile, {"summary": "new profile"})
        self.assertEqual(state.messages, [])
        self.assertEqual(state.constraints, [])
        self.assertEqual(state.shown_product_ids, set())
        self.assertEqual(state.filter_seen_recommendations(["A", "B"]), ["A", "B"])

    def test_recommendation_history_preserves_unseen_rank_order(self) -> None:
        state = StructuredSessionState("session", {})

        state.record_recommendations(["B"])

        self.assertEqual(
            state.filter_seen_recommendations(["A", "B", "C", "A"]),
            ["A", "C", "A"],
        )


if __name__ == "__main__":
    unittest.main()

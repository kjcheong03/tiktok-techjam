from __future__ import annotations

import unittest
from pathlib import Path

from baseline.query_state import CoverageAdaptiveSessionState
from scripts.run_state_transition_ablations import (
    ABLATION_VARIANT_NAMES,
    FULL_VARIANT_NAME,
    NoAmbiguousPreservationState,
    NoCompatibleAccumulationState,
    NoNoPreferencePreservationState,
    NoTargetedCorrectionState,
    TRANSITION_COMPARISON_EDGES,
    TRANSITION_POLICY_NAMES,
    default_variant_registry,
    paired_transition_comparisons,
    transition_policies,
)


class TransitionAblationStateTest(unittest.TestCase):
    def test_no_compatible_accumulation_only_drops_same_message_alternatives(self) -> None:
        full = CoverageAdaptiveSessionState("full", {})
        ablated = NoCompatibleAccumulationState("ablated", {})
        for state in (full, ablated):
            state.last_asked_attribute = "feature"
            state.observe(
                "For that, what matters is: imported; wrap closure.",
                2,
            )

        self.assertEqual(full.active_values("feature"), ["imported", "wrap closure"])
        self.assertEqual(ablated.active_values("feature"), ["wrap closure"])
        self.assertEqual(full.messages, ablated.messages)

    def test_no_targeted_correction_only_drops_unrelated_non_category_state(self) -> None:
        full = CoverageAdaptiveSessionState("full", {})
        ablated = NoTargetedCorrectionState("ablated", {})
        for state in (full, ablated):
            state.observe(
                "I'm looking for shoes. A key requirement is: black leather.",
                1,
            )
            state.record_recommendations(["shown-before-correction"])
            state.observe(
                "Actually, ignore my earlier preference. What I need is: navy.",
                2,
            )

        self.assertEqual(full.active_values("category"), ablated.active_values("category"))
        self.assertEqual(full.active_values("color"), ablated.active_values("color"))
        self.assertEqual(full.active_values("material"), ["black leather"])
        self.assertEqual(ablated.active_values("material"), [])
        self.assertEqual(full.shown_product_ids, set())
        self.assertEqual(ablated.shown_product_ids, set())

    def test_no_ambiguous_preservation_only_drops_vague_correction_targets(self) -> None:
        full = CoverageAdaptiveSessionState("full", {})
        ablated = NoAmbiguousPreservationState("ablated", {})
        for state in (full, ablated):
            state.observe(
                "I'm looking for shoes. A key requirement is: black leather.",
                1,
            )
            state.record_recommendations(["shown-before-ambiguous-correction"])
            state.observe(
                "Actually, ignore my earlier preference. What I need is: something.",
                2,
            )

        self.assertEqual(full.active_values("category"), ablated.active_values("category"))
        self.assertEqual(full.active_values("material"), ["black leather"])
        self.assertEqual(ablated.active_values("material"), [])
        self.assertEqual(
            full.shown_product_ids,
            {"shown-before-ambiguous-correction"},
        )
        self.assertEqual(
            ablated.shown_product_ids,
            {"shown-before-ambiguous-correction"},
        )

    def test_no_preference_ablation_only_hides_until_explicit_reactivation(self) -> None:
        full = CoverageAdaptiveSessionState("full", {})
        ablated = NoNoPreferencePreservationState("ablated", {})
        for state in (full, ablated):
            state.observe(
                "I'm looking for shoes. A key requirement is: leather.",
                1,
            )
            state.observe(
                "I don't have a preference for material; please use your judgment.",
                2,
            )

        self.assertIn("leather", full.build_query())
        self.assertNotIn("leather", ablated.build_query())
        self.assertEqual(full.active_values("material"), ablated.active_values("material"))
        self.assertEqual(full.no_preference_attributes, ablated.no_preference_attributes)

        for state in (full, ablated):
            state.last_asked_attribute = "material"
            state.observe("For that, what matters is: cotton.", 3)

        for state in (full, ablated):
            self.assertNotIn("material", state.no_preference_attributes)
            self.assertEqual(state.active_values("material"), ["leather", "cotton"])
            self.assertIn("leather", state.build_query())
            self.assertIn("cotton", state.build_query())


class TransitionAblationHarnessTest(unittest.TestCase):
    def test_registry_contains_full_state_and_four_orthogonal_ablations(self) -> None:
        self.assertEqual(
            list(default_variant_registry()),
            [FULL_VARIANT_NAME, *ABLATION_VARIANT_NAMES],
        )
        self.assertEqual(
            TRANSITION_COMPARISON_EDGES,
            tuple((name, FULL_VARIANT_NAME) for name in ABLATION_VARIANT_NAMES),
        )
        self.assertEqual(TRANSITION_POLICY_NAMES, ("fixed_turn_order", "fixed_other"))
        self.assertEqual(list(transition_policies()), list(TRANSITION_POLICY_NAMES))

    def test_all_factories_keep_keyword_only_history_filtered_agents(self) -> None:
        registry = default_variant_registry()
        for name, spec in registry.items():
            with self.subTest(variant=name):
                agent = spec.factory(Path("unused"), object(), transition_policies()["fixed_other"])
                self.assertIsNone(agent.dense)
                self.assertTrue(agent.filter_seen_recommendations)
                self.assertTrue(agent.stateful)

    def test_paired_comparisons_are_ablated_to_full(self) -> None:
        session = {
            "sample_id": "sample",
            "hit": True,
            "first_hit_turn": 1,
            "best_rank": 1,
        }
        variant_results = {
            name: {
                policy: {"sessions": [dict(session)]}
                for policy in TRANSITION_POLICY_NAMES
            }
            for name in [FULL_VARIANT_NAME, *ABLATION_VARIANT_NAMES]
        }

        comparisons = paired_transition_comparisons(variant_results)

        for policy in TRANSITION_POLICY_NAMES:
            self.assertEqual(
                list(comparisons[policy]),
                [f"{name} -> {FULL_VARIANT_NAME}" for name in ABLATION_VARIANT_NAMES],
            )


if __name__ == "__main__":
    unittest.main()

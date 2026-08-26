from __future__ import annotations

import unittest

from baseline.question_policy import current_order, fixed_other
from baseline.state import SessionState
from scripts.run_state_baselines import (
    DEFAULT_COMPARISON_EDGES,
    _V1PolicyAdapter,
    compare_paired_sessions,
    default_variant_registry,
)


class QuestionPolicyTest(unittest.TestCase):
    def test_current_order_delegates_to_existing_state_semantics(self) -> None:
        state = SessionState("session", {})
        self.assertEqual(current_order(state, turn=1), "material")
        state.observe("I don't have a preference for material; please use your judgment.", 2)
        self.assertEqual(current_order(state, turn=2), "color")

    def test_fixed_other_is_constant_and_does_not_inspect_state(self) -> None:
        class StateThatMustNotBeRead:
            def choose_question(self) -> str:
                raise AssertionError("fixed_other should not inspect state")

        self.assertEqual(fixed_other(StateThatMustNotBeRead(), turn=99), "other")

    def test_fixed_other_adapter_records_the_attribute_actually_asked(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.sessions = {"session": SessionState("session", {})}

            def respond(self, session_id: str, message: str, turn: int, top_k: int) -> dict:
                state = self.sessions[session_id]
                state.asked_attributes.append("material")
                state.last_asked_attribute = "material"
                return {"message": "material?", "ask_attribute": "material", "recommendations": []}

        agent = FakeAgent()
        response = _V1PolicyAdapter(agent, fixed_other).respond("session", "hello", 1, 10)

        self.assertEqual(response["ask_attribute"], "other")
        self.assertEqual(agent.sessions["session"].asked_attributes, [])
        self.assertEqual(agent.sessions["session"].last_asked_attribute, "other")


class PairedComparisonTest(unittest.TestCase):
    def test_default_registry_orders_cumulative_variants(self) -> None:
        self.assertEqual(
            list(default_variant_registry()),
            [
                "v1_keyword_state",
                "v2_state_only",
                "raw_history_no_state",
                "v2_coverage_adaptive_query",
                "v2_coverage_adaptive_history",
            ],
        )
        self.assertIn(
            ("raw_history_no_state", "v2_coverage_adaptive_query"),
            DEFAULT_COMPARISON_EDGES,
        )
        self.assertIn(
            ("v2_state_only", "v2_coverage_adaptive_query"),
            DEFAULT_COMPARISON_EDGES,
        )
        self.assertIn(
            ("v2_coverage_adaptive_query", "v2_coverage_adaptive_history"),
            DEFAULT_COMPARISON_EDGES,
        )

    def test_reports_conversion_turn_and_rank_deltas(self) -> None:
        before = {
            "sessions": [
                {
                    "sample_id": "miss-to-hit",
                    "hit": False,
                    "first_hit_turn": None,
                    "best_rank": None,
                },
                {"sample_id": "hit-to-miss", "hit": True, "first_hit_turn": 2, "best_rank": 4},
                {
                    "sample_id": "earlier-and-better",
                    "hit": True,
                    "first_hit_turn": 4,
                    "best_rank": 8,
                },
                {"sample_id": "later-and-worse", "hit": True, "first_hit_turn": 2, "best_rank": 3},
                {"sample_id": "unchanged", "hit": True, "first_hit_turn": 2, "best_rank": 3},
                {
                    "sample_id": "before-only",
                    "hit": False,
                    "first_hit_turn": None,
                    "best_rank": None,
                },
            ]
        }
        after = {
            "sessions": [
                {"sample_id": "miss-to-hit", "hit": True, "first_hit_turn": 3, "best_rank": 5},
                {
                    "sample_id": "hit-to-miss",
                    "hit": False,
                    "first_hit_turn": None,
                    "best_rank": None,
                },
                {
                    "sample_id": "earlier-and-better",
                    "hit": True,
                    "first_hit_turn": 1,
                    "best_rank": 2,
                },
                {"sample_id": "later-and-worse", "hit": True, "first_hit_turn": 3, "best_rank": 5},
                {"sample_id": "unchanged", "hit": True, "first_hit_turn": 2, "best_rank": 3},
                {"sample_id": "after-only", "hit": True, "first_hit_turn": 1, "best_rank": 1},
            ]
        }

        report = compare_paired_sessions(before, after)

        self.assertEqual(report["paired_count"], 5)
        self.assertEqual(report["unpaired_before_count"], 1)
        self.assertEqual(report["unpaired_after_count"], 1)
        self.assertEqual(report["miss_to_hit"], 1)
        self.assertEqual(report["hit_to_miss"], 1)
        self.assertEqual(report["earlier_hit_turn"], 1)
        self.assertEqual(report["later_hit_turn"], 1)
        self.assertEqual(report["better_target_rank"], 1)
        self.assertEqual(report["worse_target_rank"], 1)
        self.assertEqual(report["miss_to_hit_sample_ids"], ["miss-to-hit"])
        self.assertEqual(report["earlier_hit_turn_sample_ids"], ["earlier-and-better"])
        self.assertEqual(report["worse_target_rank_sample_ids"], ["later-and-worse"])


if __name__ == "__main__":
    unittest.main()

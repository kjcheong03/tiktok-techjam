from __future__ import annotations

import unittest

from ghostlab.state.memory import ConversationState


class HardenedStateTest(unittest.TestCase):
    def state(self) -> ConversationState:
        return ConversationState("s", {})

    def test_multiple_compatible_features_remain_active(self) -> None:
        state = self.state()
        state.observe(
            "I'm looking for hiking shoes. A key requirement is: waterproof; cushioned.",
            1,
        )
        active = [item.value for item in state.active_values()]
        self.assertIn("waterproof", active)
        self.assertIn("cushioned", active)

    def test_color_replacement_is_scoped(self) -> None:
        state = self.state()
        state.observe(
            "I'm looking for shoes. A key requirement is: black; waterproof.", 1
        )
        state.observe("Actually, what I need is: navy.", 2)
        active = [item.value for item in state.active_values()]
        self.assertNotIn("black", active)
        self.assertIn("waterproof", active)
        self.assertIn("navy", active)

    def test_earlier_preference_override_clears_non_category_preferences(self) -> None:
        state = self.state()
        state.observe(
            "I'm looking for shoes. A key requirement is: black; waterproof.", 1
        )
        state.observe(
            "Actually, ignore my earlier preference. What I need is: cushioned.", 3
        )
        active = [item.value for item in state.active_values()]
        self.assertIn("shoes", active)
        self.assertIn("cushioned", active)
        self.assertNotIn("black", active)
        self.assertNotIn("waterproof", active)

    def test_global_reset_clears_positive_and_negative_then_ingests_current(
        self,
    ) -> None:
        state = self.state()
        state.observe(
            "I'm looking for shoes. A key requirement is: black; waterproof.", 1
        )
        state.observe("Avoid leather.", 2)
        state.observe("Start over. What I need is: cotton.", 3)
        self.assertEqual([item.value for item in state.active_values()], ["cotton"])
        self.assertEqual(state.active_values("negative"), [])

    def test_targeted_correction_preserves_unrelated_preferences(self) -> None:
        state = self.state()
        state.observe(
            "I'm looking for shoes. A key requirement is: black; waterproof.", 1
        )
        state.observe("Actually, what I need is: navy.", 2)
        active = [item.value for item in state.active_values()]
        self.assertIn("waterproof", active)
        self.assertIn("navy", active)
        self.assertNotIn("black", active)

    def test_ambiguous_actually_preserves_unrelated_state(self) -> None:
        state = self.state()
        state.observe(
            "I'm looking for shoes. A key requirement is: black; waterproof.", 1
        )
        state.observe("Actually, let me think about that.", 2)
        active = [item.value for item in state.active_values()]
        self.assertIn("black", active)
        self.assertIn("waterproof", active)

    def test_repeated_slot_correction_leaves_only_latest_value(self) -> None:
        state = self.state()
        state.observe("I'm looking for shoes. A key requirement is: black.", 1)
        state.observe("Actually, what I need is: navy.", 2)
        state.observe("Actually, what I need is: red.", 3)
        colors = [
            item.value for item in state.active_values() if item.attribute == "color"
        ]
        self.assertEqual(colors, ["red"])

    def test_negation_invalidates_only_matching_value(self) -> None:
        state = self.state()
        state.observe(
            "I'm looking for shoes. A key requirement is: black; waterproof.", 1
        )
        state.observe("Avoid black.", 2)
        active = [item.value for item in state.active_values()]
        self.assertNotIn("black", active)
        self.assertIn("waterproof", active)

    def test_category_override_preserves_unscoped_budget(self) -> None:
        state = self.state()
        state.observe(
            "I'm looking for bags. A key requirement is: leather; budget under $50.", 1
        )
        state.observe("Actually, I need shoes instead.", 2)
        active = [item.value for item in state.active_values()]
        self.assertNotIn("bags", active)
        self.assertNotIn("leather", active)
        self.assertIn("budget under $50", active)
        self.assertIn("shoes", active)

    def test_negative_and_no_preference_are_remembered(self) -> None:
        state = self.state()
        state.observe("I'm looking for shirts. A key requirement is: black.", 1)
        state.observe("Actually, not black.", 2)
        state.observe("I don't have an additional preference for material.", 3)
        self.assertNotIn("black", [item.value for item in state.active_values()])
        self.assertNotEqual(state.choose_question(), "material")

    def test_sessions_do_not_share_state(self) -> None:
        first, second = self.state(), ConversationState("other", {})
        first.observe("I'm looking for shoes.", 1)
        self.assertIsNone(second.active_category)


if __name__ == "__main__":
    unittest.main()

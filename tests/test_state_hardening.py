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

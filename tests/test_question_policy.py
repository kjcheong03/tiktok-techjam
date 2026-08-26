from __future__ import annotations

import unittest

from baseline.question_policy import (
    QUESTION_POLICIES,
    current_order,
    fixed_other,
    fixed_turn_order,
)


class QuestionPolicyTest(unittest.TestCase):
    def test_fixed_turn_order_has_exact_original_sequence(self) -> None:
        class StateThatMustNotBeRead:
            def __getattribute__(self, name: str) -> object:
                raise AssertionError(f"state should not be read: {name}")

        state = StateThatMustNotBeRead()
        self.assertEqual(
            [fixed_turn_order(state, turn) for turn in range(1, 9)],
            ["material", "color", "style", "use_case", "feature", "budget", "size", None],
        )

    def test_fixed_turn_order_requires_a_concrete_turn(self) -> None:
        with self.assertRaisesRegex(ValueError, "concrete integer turn"):
            fixed_turn_order(None, None)  # type: ignore[arg-type]

    def test_registry_contains_three_distinct_policies(self) -> None:
        self.assertEqual(
            set(QUESTION_POLICIES),
            {"current_order", "fixed_other", "fixed_turn_order"},
        )
        self.assertIs(QUESTION_POLICIES["current_order"], current_order)
        self.assertIs(QUESTION_POLICIES["fixed_other"], fixed_other)
        self.assertIs(QUESTION_POLICIES["fixed_turn_order"], fixed_turn_order)
        self.assertEqual(len({id(policy) for policy in QUESTION_POLICIES.values()}), 3)


if __name__ == "__main__":
    unittest.main()

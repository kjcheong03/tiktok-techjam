from __future__ import annotations

import unittest

from ghostlab.evaluation.reward_deltas import (
    efficiency_at_turn,
    swap_reward_delta,
    terminal_session_reward,
)


class RewardDeltaTests(unittest.TestCase):
    def test_exact_organizer_terms(self) -> None:
        self.assertAlmostEqual(efficiency_at_turn(1), 1.0)
        self.assertAlmostEqual(efficiency_at_turn(10), 0.1)
        self.assertAlmostEqual(terminal_session_reward(1, 1), 1.0)
        self.assertAlmostEqual(terminal_session_reward(10, 10), 0.55)
        self.assertEqual(terminal_session_reward(11, 1), 0.0)
        self.assertEqual(terminal_session_reward(None, 11), 0.0)

    def test_rank_ten_boundary_includes_hit_and_efficiency(self) -> None:
        early = swap_reward_delta(10, 11, 1)
        late = swap_reward_delta(10, 11, 10)
        self.assertAlmostEqual(early, 0.73)
        self.assertAlmostEqual(late, 0.55)
        self.assertGreater(early, late)

    def test_inside_top_ten_only_changes_reciprocal_rank(self) -> None:
        self.assertAlmostEqual(swap_reward_delta(1, 2, 1), 0.15)
        self.assertAlmostEqual(
            swap_reward_delta(1, 2, 1), swap_reward_delta(1, 2, 10)
        )

    def test_invalid_coordinates_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            terminal_session_reward(0, 1)
        with self.assertRaises(ValueError):
            efficiency_at_turn(12)


if __name__ == "__main__":
    unittest.main()

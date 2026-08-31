from __future__ import annotations

import unittest

from ghostlab.runtime.component_fallback import ComponentFallback


class ComponentFallbackTests(unittest.TestCase):
    def test_falls_back_for_small_or_constraint_losing_result(self) -> None:
        fallback = ComponentFallback(minimum_results=2)
        self.assertTrue(fallback.choose(["A"], ["B", "C"]).used_fallback)
        decision = fallback.choose(
            ["A", "B"],
            ["B", "C"],
            candidate_constraint_coverage={"color": 0.2},
            base_constraint_coverage={"color": 0.8},
        )
        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.reason, "constraint_coverage:color")

    def test_accepts_sound_component_and_deduplicates(self) -> None:
        decision = ComponentFallback(minimum_results=2).choose(
            ["A", "A", "B"], ["C", "D"]
        )
        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.ranking, ("A", "B"))


if __name__ == "__main__":
    unittest.main()

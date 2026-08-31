from __future__ import annotations

import unittest

from ghostlab.research.counterfactual import ActionOutcome, CounterfactualEvaluator


class CounterfactualTest(unittest.TestCase):
    def test_known_synthetic_optimum_is_recovered(self) -> None:
        outcomes = [
            ActionOutcome("s", "color", 0.2, False, None, None),
            ActionOutcome("s", "material", 0.8, True, 2, 1),
            ActionOutcome("s", None, 0.1, False, None, None),
        ]
        self.assertEqual(CounterfactualEvaluator.best(outcomes).action, "material")

    def test_ties_are_independent_of_input_order(self) -> None:
        first = ActionOutcome("s", "style", 0.5, True, 2, 2)
        second = ActionOutcome("s", "color", 0.5, True, 2, 2)
        self.assertEqual(
            CounterfactualEvaluator.best([first, second]),
            CounterfactualEvaluator.best([second, first]),
        )


if __name__ == "__main__":
    unittest.main()

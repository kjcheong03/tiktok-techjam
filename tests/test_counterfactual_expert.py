from __future__ import annotations

import unittest

from ghostlab.research.counterfactual_expert import (
    CounterfactualExpert,
    ExpertState,
    aggregate_expert_iterations,
)


class CounterfactualExpertTests(unittest.TestCase):
    def test_selects_best_penalized_action_deterministically(self) -> None:
        expert = CounterfactualExpert(("base", "expensive"), {"expensive": 0.02})
        label = expert.label(
            ExpertState("s", 1, {"turn": 0.1}, {"base": 0.5, "expensive": 0.51})
        )
        self.assertEqual(label.best_action_id, "base")

    def test_aggregation_is_bounded_and_deduplicated(self) -> None:
        expert = CounterfactualExpert(("a",), {})
        label = expert.label(ExpertState("s", 1, {"x": 0.0}, {"a": 1.0}))
        merged = aggregate_expert_iterations([[label], [label], [label]], maximum_rounds=2)
        self.assertEqual(merged, [label])

    def test_rejects_forbidden_runtime_features(self) -> None:
        expert = CounterfactualExpert(("a",), {})
        with self.assertRaisesRegex(ValueError, "runtime boundary"):
            expert.label(ExpertState("s", 1, {"target": 1.0}, {"a": 1.0}))


if __name__ == "__main__":
    unittest.main()

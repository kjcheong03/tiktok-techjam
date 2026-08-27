from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from ghostlab.optimization.conditional import (
    ConditionalSearchSpace,
    TuningContext,
    suggest_for_combination,
    two_way_simplex_grid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConditionalHPOTests(unittest.TestCase):
    def setUp(self) -> None:
        value = json.loads(
            (PROJECT_ROOT / "configs/search/wave2_weight_space_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.space = ConditionalSearchSpace.model_validate(value)

    def test_only_combination_relevant_weights_are_exposed(self) -> None:
        parameters = self.space.for_techniques(
            ("question.candidate_eig.v1", "ranking.reward_lambdamart.v1")
        )
        self.assertEqual(
            {item.name for item in parameters},
            {"eig_candidate_k", "question_value_margin", "rerank_k"},
        )

    def test_suggestion_requires_inner_validation_context(self) -> None:
        with self.assertRaises(ValidationError):
            TuningContext(outer_fold=0, inner_fold=0, split_role="f3")  # type: ignore[arg-type]
        suggestion = suggest_for_combination(
            self.space,
            ("question.candidate_eig.v1",),
            (),
            context=TuningContext(outer_fold=0, inner_fold=1),
            seed=17,
        )
        self.assertEqual(
            {name for name, _ in suggestion},
            {"eig_candidate_k", "question_value_margin"},
        )

    def test_simplex_grid_is_normalized(self) -> None:
        grid = two_way_simplex_grid(0.25)
        self.assertEqual(len(grid), 5)
        self.assertTrue(all(abs(left + right - 1.0) < 1e-12 for left, right in grid))
        with self.assertRaises(ValueError):
            two_way_simplex_grid(0.3)


if __name__ == "__main__":
    unittest.main()

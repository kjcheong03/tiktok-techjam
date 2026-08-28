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
from ghostlab.research.technique_suite import UnifiedTechniqueConfig

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

    def test_adaptive_space_exposes_real_runtime_parameters(self) -> None:
        value = json.loads(
            (
                PROJECT_ROOT / "configs/search/adaptive_parameter_space_v1.json"
            ).read_text(encoding="utf-8")
        )
        space = ConditionalSearchSpace.model_validate(value)
        unknown = {item.name for item in space.parameters} - {
            *UnifiedTechniqueConfig.model_fields,
            "fusion_sparse_share",
            "question_order_id",
            "sparse_title_weight",
            "sparse_categories_weight",
            "sparse_features_weight",
            "sparse_details_weight",
            "sparse_store_weight",
            "sparse_description_weight",
        }
        self.assertEqual(unknown, set())
        techniques = ("retrieval.e5", "fusion.weighted")
        parameters = space.for_techniques(techniques)
        names = {item.name for item in parameters}
        self.assertEqual(
            names,
            {
                "retrieval_k",
                "fusion_sparse_share",
                "dense_activation",
                "dense_activation_min_entropy",
            },
        )
        suggestion = suggest_for_combination(
            space,
            techniques,
            (),
            context=TuningContext(outer_fold=0, inner_fold=1),
            seed=11,
        )
        self.assertEqual({name for name, _ in suggestion}, names)

    def test_residual_search_space_is_conditional_and_complete(self) -> None:
        space = ConditionalSearchSpace.model_validate_json(
            (
                PROJECT_ROOT / "configs/search/adaptive_parameter_space_v1.json"
            ).read_text(encoding="utf-8")
        )
        names = {
            item.name
            for item in space.for_techniques(
                ("retrieval.sparse", "ranking.top10_residual_reranker.v2")
            )
        }
        residual_names = {name for name in names if name.startswith("residual_")}
        self.assertEqual(
            residual_names,
            {
                "residual_feature_set",
                "residual_model_variant",
                "residual_regularization",
                "residual_rerank_depth",
                "residual_model_weight",
                "residual_minimum_expected_gain",
                "residual_minimum_probability_margin",
                "residual_maximum_moved_ids",
            },
        )
        without_residual = {
            item.name
            for item in space.for_techniques(("retrieval.sparse",))
        }
        self.assertFalse(any(name.startswith("residual_") for name in without_residual))


if __name__ == "__main__":
    unittest.main()

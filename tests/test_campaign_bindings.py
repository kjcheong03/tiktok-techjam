from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghostlab.campaign.bindings import (
    BindingConflictError,
    default_binding_registry,
)
from ghostlab.campaign.models import CandidateSpec
from ghostlab.research.technique_suite import load_suite_config

ROOT = Path(__file__).resolve().parents[1]


def _candidate(
    *techniques: str,
    parameters: tuple[tuple[str, str | int | float | bool], ...] = (),
) -> CandidateSpec:
    return CandidateSpec(
        candidate_id="binding-test",
        baseline_id="unified_keyword_research_v1",
        techniques=techniques,
        parameters=parameters,
        complexity=len(techniques),
        generation="single" if len(techniques) == 1 else "pair",
    )


def test_registry_covers_every_declared_wave1_and_wave2_id() -> None:
    declared = set()
    for name in ("catalog_v1.json", "catalog_v2.json"):
        payload = json.loads((ROOT / "configs/techniques" / name).read_text())
        declared.update(item["id"] for item in payload["techniques"])
    registry = default_binding_registry()
    assert set(registry.bindings) == declared
    assert all(binding.reason for binding in registry.bindings.values())


def test_materialization_preserves_unpatched_assets_and_applies_parameters() -> None:
    base = load_suite_config(ROOT / "configs/suites/keyword_research.json")
    registry = default_binding_registry()
    config = registry.materialize(
        base,
        _candidate(
            "question.candidate_eig.v1",
            parameters=(("eig_candidate_k", 200),),
        ),
    )
    assert config.experiment_id == "binding-test"
    assert config.question_variant == "candidate_eig"
    assert config.eig_candidate_k == 200
    assert config.reranker == base.reranker
    assert config.reranker_model_asset == base.reranker_model_asset


def test_composable_assets_are_preserved_verbatim() -> None:
    registry = default_binding_registry()
    binding = registry.bindings["ranking.reward_lambdamart.v1"]
    assert binding.asset_paths == (
        "artifacts/models/w2_ranking_v1/reward_lambdamart_v1.json",
    )
    config = registry.materialize_from_suite(
        ROOT / "configs/suites/keyword_research.json",
        _candidate("ranking.reward_lambdamart.v1"),
    )
    assert config.reranker == "reward_lambdamart"
    assert config.reranker_model_asset == binding.asset_paths[0]


def test_conflicting_question_and_ranker_patches_are_rejected() -> None:
    base = load_suite_config(ROOT / "configs/suites/keyword_research.json")
    registry = default_binding_registry()
    with pytest.raises(BindingConflictError, match="question_variant"):
        registry.materialize(
            base,
            _candidate("question.candidate_eig.v1", "policy.joint_observable.v1"),
        )
    with pytest.raises(BindingConflictError, match="reranker"):
        registry.materialize(
            base,
            _candidate("ranking.reward_lambdamart.v1", "ranking.fold_ensemble.v1"),
        )


def test_dependencies_and_non_runtime_entries_fail_explicitly() -> None:
    base = load_suite_config(ROOT / "configs/suites/keyword_research.json")
    registry = default_binding_registry()
    with pytest.raises(ValueError, match="requires"):
        registry.materialize(base, _candidate("state.confidence_gated_constraints.v1"))
    with pytest.raises(ValueError, match="unavailable"):
        registry.materialize(base, _candidate("retrieval.splade_rescue.v1"))
    with pytest.raises(ValueError, match="research_only"):
        registry.materialize(base, _candidate("search.bohb.v1"))
    with pytest.raises(ValueError, match="anchor_only"):
        registry.materialize(base, _candidate("ranking.constraint_gbdt"))


def test_normalizer_and_confidence_gate_materialize_together() -> None:
    registry = default_binding_registry()
    config = registry.materialize_from_suite(
        ROOT / "configs/suites/keyword_research.json",
        _candidate(
            "state.catalog_normalizer.v1",
            "state.confidence_gated_constraints.v1",
        ),
    )
    assert config.normalizer == "catalog_v1"
    assert config.normalizer_asset == "artifacts/assets/catalog_ontology_v1.json"
    assert config.constraint_confidence == 0.9


def test_derived_fusion_share_is_normalized() -> None:
    registry = default_binding_registry()
    base = load_suite_config(ROOT / "configs/suites/keyword_research.json")
    config = registry.materialize(
        base,
        _candidate(
            "retrieval.e5",
            "fusion.weighted",
            parameters=(("fusion_sparse_share", 0.61),),
        ),
    )
    assert config.sparse_weight == pytest.approx(0.61)
    assert config.dense_weight == pytest.approx(0.39)
    assert config.sparse_weight + config.dense_weight == pytest.approx(1.0)


def test_derived_sparse_fields_and_question_order_are_typed() -> None:
    registry = default_binding_registry()
    base = load_suite_config(ROOT / "configs/suites/keyword_research.json")
    config = registry.materialize(
        base,
        _candidate(
            "question.fixed",
            "retrieval.sparse",
            parameters=(
                ("question_order_id", "need_first"),
                ("sparse_title_weight", 7.0),
                ("sparse_categories_weight", 5.0),
                ("sparse_features_weight", 3.0),
                ("sparse_details_weight", 2.0),
                ("sparse_store_weight", 1.0),
                ("sparse_description_weight", 0.5),
            ),
        ),
    )
    assert config.question_order[:3] == ("use_case", "other", "size")
    assert config.sparse_weights == (7.0, 5.0, 3.0, 2.0, 1.0, 0.5)


def test_sparse_field_weights_must_be_complete() -> None:
    registry = default_binding_registry()
    base = load_suite_config(ROOT / "configs/suites/keyword_research.json")
    with pytest.raises(ValueError, match="all six"):
        registry.materialize(
            base,
            _candidate(
                "retrieval.sparse",
                parameters=(("sparse_title_weight", 7.0),),
            ),
        )


def test_residual_parameters_materialize_without_prefitted_asset() -> None:
    registry = default_binding_registry()
    base = load_suite_config(ROOT / "configs/suites/keyword_research.json")
    config = registry.materialize(
        base,
        _candidate(
            "ranking.top10_residual_reranker.v2",
            parameters=(
                ("residual_feature_set", "metadata"),
                ("residual_model_variant", "ensemble_logistic_gbdt_d3_lr01"),
                ("residual_regularization", 0.5),
                ("residual_rerank_depth", 5),
                ("residual_model_weight", 0.75),
                ("residual_minimum_expected_gain", 0.01),
                ("residual_minimum_probability_margin", 0.02),
                ("residual_maximum_moved_ids", 4),
            ),
        ),
    )

    assert config.residual_reranker_enabled
    assert config.residual_model_asset is None
    assert config.residual_feature_set == "metadata"
    assert config.residual_model_variant == "ensemble_logistic_gbdt_d3_lr01"
    assert config.residual_regularization == 0.5
    assert config.residual_rerank_depth == 5
    assert config.residual_model_weight == 0.75

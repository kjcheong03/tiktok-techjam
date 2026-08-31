from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CandidateSpec
from ghostlab.optimization.adaptive_hybrid import AdaptiveArchitectureAudit
from ghostlab.optimization.adaptive_techniques import AdaptiveTechniqueRegistry
from ghostlab.runtime.adaptive_config import (
    AdaptiveExtensionsConfig,
    AdaptiveHybridConfig,
)

ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_CONTROL_ONLY = {
    "policy.joint_observable.v1",
    "query.expansion_guard.v1",
    "query.structured",
    "question.adaptive_heuristic",
    "question.fixed",
    "question.learned_linear",
    "question.other_always",
    "ranking.constraint_gbdt",
    "ranking.deep_dense_gbdt",
    "ranking.mmr_early.v1",
    "ranking.neural_gbdt",
    "ranking.pairwise_linear",
    "ranking.top10_residual_reranker.v2",
    "retrieval.minilm",
    "routing.decision_list",
    "routing.joint_route.v1",
    "routing.observable_stump",
    "routing.route_table",
    "state.attribute_ontology.v1",
    "state.catalog_normalizer.v1",
    "state.compressed",
    "state.confidence_gated_constraints.v1",
    "state.current",
    "state.multi",
    "state.raw_history",
}
ABSORBED = {
    "query.expansion_guard.v1": "query.catalog_prf.v1",
    "ranking.mmr_early.v1": "ranking.facet_diversity.v1",
    "state.attribute_ontology.v1": "state.catalog_normalizer.v1",
    "state.confidence_gated_constraints.v1": "state.catalog_normalizer.v1",
}
ADAPTED = {
    "ranking.top10_residual_reranker.v2",
    "retrieval.minilm",
    "state.catalog_normalizer.v1",
}


def _registry() -> AdaptiveTechniqueRegistry:
    return AdaptiveTechniqueRegistry.from_catalog(
        load_catalog(ROOT / "configs/techniques/catalog_v2.json"),
        project_root=ROOT,
    )


def _candidate(registry: AdaptiveTechniqueRegistry, *additions: str) -> CandidateSpec:
    baseline = AdaptiveHybridConfig()
    return CandidateSpec(
        candidate_id="adapted-control-test",
        baseline_id=baseline.policy_id,
        techniques=(*registry.inventory().compulsory, *additions),
        complexity=len(additions),
        generation="single",
    )


def test_all_25_former_controls_have_an_explicit_adaptive_disposition() -> None:
    registry = _registry()
    remaining = ORIGINAL_CONTROL_ONLY - ADAPTED

    assert ADAPTED <= set(registry.inventory().promotable)
    assert remaining <= set(registry.inventory().control_only)
    assert {
        technique_id: registry.bindings[technique_id].absorbed_by
        for technique_id in ABSORBED
    } == ABSORBED
    assert all(
        registry.bindings[technique_id].control_class == "replacement_only"
        and registry.bindings[technique_id].reason.startswith(
            "replacement-only control:"
        )
        for technique_id in remaining - set(ABSORBED)
    )


def test_catalog_normalizer_materializes_as_a_pinned_state_v2_addition() -> None:
    registry = _registry()
    baseline = AdaptiveHybridConfig()
    config = registry.materialize(
        baseline,
        _candidate(registry, "state.catalog_normalizer.v1"),
    )
    ontology = ROOT / "artifacts/assets/catalog_ontology_v1.json"

    assert config.state.component == "state_v2"
    assert config.state.catalog_normalizer_enabled
    assert config.state.catalog_ontology_path == (
        "artifacts/assets/catalog_ontology_v1.json"
    )
    assert (
        config.state.catalog_ontology_sha256
        == hashlib.sha256(ontology.read_bytes()).hexdigest()
    )
    assert config.state.constraint_normalization_confidence == 0.9


def test_minilm_materializes_as_an_auxiliary_view_without_replacing_e5() -> None:
    registry = _registry()
    baseline = AdaptiveHybridConfig()
    candidate = _candidate(registry, "retrieval.minilm")

    registry.validate_candidate(candidate)
    config = registry.materialize(baseline, candidate)

    assert config.browsing.component == "diverse_e5_multiview"
    assert config.extensions.minilm_dense_view_enabled
    assert config.extensions.minilm_dense_retrieval_k == 80
    assert config.extensions.minilm_dense_weight == 0.15
    assert registry.catalog.techniques["retrieval.e5"].exclusive_group == (
        "dense_backend"
    )
    assert registry.catalog.techniques["retrieval.minilm"].exclusive_group is None


def test_residual_config_accepts_unresolved_fresh_fit_but_pairs_asset_hashes() -> None:
    unresolved = AdaptiveExtensionsConfig(top10_residual_enabled=True)
    assert unresolved.top10_residual_model_path is None
    assert unresolved.top10_residual_fit_receipt_path is None

    with pytest.raises(ValueError, match="residual model path and SHA256"):
        AdaptiveExtensionsConfig(top10_residual_model_path="model.joblib")
    with pytest.raises(ValueError, match="residual fit receipt path and SHA256"):
        AdaptiveExtensionsConfig(top10_residual_fit_receipt_sha256="0" * 64)


@pytest.mark.parametrize(
    ("technique_id", "auxiliary_backend"),
    (
        ("ranking.fixed_lexical", "fixed_lexical"),
        ("ranking.metadata_gbdt", "gbdt"),
        ("ranking.reward_lambdamart.v1", "gbdt"),
        ("ranking.turn_aware_lambdamart.v1", "gbdt"),
        ("ranking.fold_ensemble.v1", "rank_ensemble"),
    ),
)
def test_historical_rankers_are_bounded_additions_not_union_replacements(
    technique_id: str,
    auxiliary_backend: str,
) -> None:
    registry = _registry()
    baseline = AdaptiveHybridConfig()
    config = registry.materialize(baseline, _candidate(registry, technique_id))

    assert config.union_ranker.backend == baseline.union_ranker.backend == "gbdt"
    assert config.union_ranker.model_path == baseline.union_ranker.model_path
    assert config.union_ranker.model_sha256 == baseline.union_ranker.model_sha256
    assert config.union_ranker.auxiliary_technique_id == technique_id
    assert config.union_ranker.auxiliary_backend == auxiliary_backend
    assert 0.0 < config.union_ranker.auxiliary_weight <= 0.25
    assert config.union_ranker.auxiliary_rerank_k <= 200


def test_rank_stack_is_a_bounded_auxiliary_over_its_fold_dependency() -> None:
    registry = _registry()
    baseline = AdaptiveHybridConfig()
    candidate = _candidate(
        registry,
        "fusion.rank_stack.v1",
        "ranking.fold_ensemble.v1",
    )
    config = registry.materialize(baseline, candidate)

    assert config.union_ranker.backend == "gbdt"
    assert config.union_ranker.model_path == baseline.union_ranker.model_path
    assert config.union_ranker.auxiliary_technique_id == "fusion.rank_stack.v1"
    assert config.union_ranker.auxiliary_backend == "rank_ensemble"
    assert config.union_ranker.auxiliary_model_path == (
        "artifacts/models/w2_ranking_v1/rank_stack.json"
    )


def test_architecture_audit_rejects_primary_union_replacement() -> None:
    baseline = AdaptiveHybridConfig()
    replaced = baseline.model_copy(
        update={
            "union_ranker": baseline.union_ranker.model_copy(
                update={
                    "backend": "deterministic",
                    "model_path": None,
                    "model_sha256": None,
                }
            )
        }
    )
    with pytest.raises(ValueError, match="compulsory source-aware union GBDT"):
        AdaptiveArchitectureAudit.validate(replaced)


def test_weighted_fusion_candidate_is_not_an_identity_patch() -> None:
    registry = _registry()
    baseline = AdaptiveHybridConfig()
    config = registry.materialize(
        baseline,
        _candidate(registry, "fusion.weighted"),
    )

    assert config.merger.strategy == "weighted"
    assert config.merger.buying_keyword_weight == 0.8
    assert config.merger.browsing_vector_weight == 0.7
    assert config.merger != baseline.merger


def test_replacement_only_control_fails_closed_with_specific_reason() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="replacement-only control"):
        registry.materialize(
            AdaptiveHybridConfig(),
            _candidate(registry, "routing.decision_list"),
        )

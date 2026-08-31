from __future__ import annotations

from pathlib import Path

import pytest

from ghostlab.campaign.catalog import load_catalog
from ghostlab.campaign.models import CandidateSpec
from ghostlab.optimization.adaptive_campaign import (
    AdaptiveEvaluation,
    AdaptiveGhostLabEngine,
)
from ghostlab.optimization.adaptive_hybrid import AdaptiveArchitectureAudit
from ghostlab.optimization.adaptive_techniques import AdaptiveTechniqueRegistry
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig

ROOT = Path(__file__).resolve().parents[1]


def _engine() -> AdaptiveGhostLabEngine:
    catalog = load_catalog(ROOT / "configs/techniques/catalog_v2.json")
    registry = AdaptiveTechniqueRegistry.from_catalog(catalog, project_root=ROOT)
    baseline = AdaptiveHybridConfig()
    baseline = baseline.model_copy(
        update={
            "semantic_ranker": baseline.semantic_ranker.model_copy(
                update={
                    "backend": "local_causal_relevance",
                    "model_id": "smollm2-1.7b-instruct",
                    "weight": 0.05,
                    "rerank_k": 10,
                }
            )
        }
    )
    return AdaptiveGhostLabEngine(
        baseline=baseline,
        registry=registry,
        candidate_limit=500,
        beam_width=24,
    )


def test_adaptive_registry_is_exhaustive_and_preserves_every_catalog_record() -> None:
    engine = _engine()
    inventory = engine.registry.inventory()
    classified = (
        inventory.compulsory
        + inventory.promotable
        + inventory.control_only
        + inventory.research_only
        + inventory.unavailable
    )
    assert inventory.total == len(engine.registry.catalog.techniques)
    assert len(classified) == inventory.total
    assert len(set(classified)) == inventory.total
    assert "state.baseline_v2" in inventory.compulsory
    assert "retrieval.sparse" in inventory.compulsory
    assert "retrieval.e5" in inventory.compulsory
    assert "prior.quality" in inventory.promotable
    assert "query.catalog_prf.v1" in inventory.promotable
    assert "ranking.metadata_gbdt" in inventory.promotable
    assert "state.current" in inventory.control_only
    assert "search.multifidelity_racing" in inventory.research_only


def test_every_initial_challenger_materializes_without_changing_architecture() -> None:
    engine = _engine()
    plan = engine.initial_plan()
    assert plan.candidates
    assert plan.candidates[0].generation == "control"
    assert {
        technique
        for candidate in plan.candidates
        for technique in candidate.techniques
        if technique in engine.registry.inventory().promotable
    } == set(engine.registry.inventory().promotable)
    for candidate in plan.candidates:
        config = engine.materialize(candidate)
        assert AdaptiveArchitectureAudit.validate(config) is config
        assert config.architecture == "adaptive_hybrid_1a_3b_v1"
        union_implementations = set(candidate.techniques) & {
            "ranking.fixed_lexical",
            "ranking.metadata_gbdt",
            "ranking.reward_lambdamart.v1",
            "ranking.turn_aware_lambdamart.v1",
            "ranking.fold_ensemble.v1",
        }
        assert len(union_implementations) <= 1


def test_semantic_lane_uses_only_the_fixed_weight_and_depth_schedule() -> None:
    engine = _engine()
    calibration = [
        candidate
        for candidate in engine.initial_plan().candidates
        if candidate.candidate_id.startswith("semantic-calibration-")
    ]
    assert {
        (
            dict(candidate.parameters)["semantic_weight"],
            dict(candidate.parameters)["semantic_rerank_k"],
        )
        for candidate in calibration
    } == {(0.10, 10), (0.15, 10), (0.20, 10)}
    assert engine.baseline.semantic_ranker.rerank_k == 10


def test_compulsory_or_control_techniques_cannot_be_toggled_as_additions() -> None:
    engine = _engine()
    mandatory = engine.registry.inventory().compulsory
    candidate = CandidateSpec(
        candidate_id="illegal-state-control",
        baseline_id=engine.baseline.policy_id,
        techniques=(*mandatory, "state.current"),
        complexity=1,
        generation="single",
    )
    with pytest.raises(ValueError, match="not submission-promotable"):
        engine.materialize(candidate)


def test_optional_patches_are_real_and_conditionally_tunable() -> None:
    engine = _engine()
    mandatory = engine.registry.inventory().compulsory
    candidate = CandidateSpec(
        candidate_id="quality-facet-rrf",
        baseline_id=engine.baseline.policy_id,
        techniques=(
            *mandatory,
            "fusion.rrf",
            "prior.quality",
            "query.catalog_prf.v1",
            "ranking.facet_diversity.v1",
        ),
        complexity=4,
        generation="beam",
    )
    config = engine.materialize(candidate)
    assert config.merger.strategy == "rrf"
    assert config.extensions.quality_prior_weight == 0.2
    assert config.extensions.query_prf_enabled
    assert config.extensions.facet_diversity_enabled
    tuned = engine.hpo_candidate(
        candidate,
        (
            ("merger_rrf_constant", 25),
            ("quality_prior_weight", 0.15),
            ("query_prf_feedback_k", 8),
            ("facet_relevance_weight", 0.7),
        ),
        ordinal=1,
    )
    tuned_config = engine.materialize(tuned)
    assert tuned_config.merger.rrf_constant == 25
    assert tuned_config.extensions.quality_prior_weight == 0.15
    assert tuned_config.extensions.query_prf_feedback_k == 8
    assert tuned_config.extensions.facet_relevance_weight == 0.7
    with pytest.raises(ValueError, match="inactive"):
        engine.hpo_candidate(
            engine.incumbent,
            (("quality_prior_weight", 0.2),),
            ordinal=2,
        )


def test_engine_has_no_six_technique_champion_cap_and_runs_all_racing_stages() -> None:
    engine = _engine()
    assert len(engine.registry.inventory().promotable) > 6
    assert engine.limits.max_order == len(engine.registry.inventory().promotable)

    def evaluator(config, candidate, fidelity):  # type: ignore[no-untyped-def]
        AdaptiveArchitectureAudit.validate(config)
        gain = (
            0.0
            if candidate.generation == "control"
            else 0.03 + 0.002 * candidate.complexity
        )
        return AdaptiveEvaluation(
            candidate_id=candidate.candidate_id,
            fidelity=fidelity,
            score=0.5 + gain,
            session_rewards=(0.4 + gain,) * 8,
            behavior_novelty=0.1 * bool(candidate.complexity),
            fit_verified=True,
        )

    result = engine.run(
        evaluator,
        f1_candidates=8,
        f2_candidates=4,
        higher_order_rounds=1,
        hpo_trials_per_structure=1,
    )
    assert result.stages["f0"]
    assert result.stages["f1"]
    assert result.stages["f2"]
    assert any(
        "-hpo-" in record.candidate.candidate_id for record in result.stages["f1"]
    )
    assert result.promoted
    assert result.selected.generation != "control"
    assert result.selected_semantic_weight in {0.05, 0.10, 0.15, 0.20}
    assert result.selected_semantic_depth == 10
    assert all(
        AdaptiveArchitectureAudit.validate(engine.materialize(record.candidate))
        for records in result.stages.values()
        for record in records
    )


def test_quality_gate_rejects_a_hit_at_10_regression() -> None:
    engine = _engine()
    challenger = engine.semantic_candidate(weight=0.10, depth=10, suffix="gate")

    def evaluator(config, candidate, fidelity):  # type: ignore[no-untyped-def]
        AdaptiveArchitectureAudit.validate(config)
        is_control = candidate.generation == "control"
        return AdaptiveEvaluation(
            candidate_id=candidate.candidate_id,
            fidelity=fidelity,
            score=0.8,
            session_rewards=(0.8,) * 8,
            hit_rate_at_10=0.90 if is_control else 0.89,
            mrr=0.70 if is_control else 0.71,
        )

    records = engine._evaluate_stage((engine.incumbent, challenger), "f0", evaluator)
    challenger_record = next(
        item
        for item in records
        if item.candidate.candidate_id == challenger.candidate_id
    )
    assert challenger_record.decision == "REJECT"
    assert any(
        failure.startswith("hit_at_10_regression:")
        for failure in challenger_record.gate_failures
    )

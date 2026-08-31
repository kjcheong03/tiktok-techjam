from __future__ import annotations

from pathlib import Path

from ghostlab.campaign.catalog import load_catalog
from ghostlab.optimization.adaptive_campaign import (
    AdaptiveEvaluation,
    AdaptiveGhostLabEngine,
)
from ghostlab.optimization.adaptive_techniques import AdaptiveTechniqueRegistry
from ghostlab.optimization.adaptive_warm_start import load_adaptive_warm_start
from ghostlab.runtime.adaptive_factory import load_adaptive_hybrid_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/adaptive_hybrid_1a_3b_1650_final_v1_selected.json"
WARM_START = "configs/warm_starts/adaptive_d4e040a07e6d_to_1a_3b_v1.json"


def _loaded() -> tuple[AdaptiveTechniqueRegistry, object, object, object]:
    baseline = load_adaptive_hybrid_config(ROOT / CONFIG)
    catalog = load_catalog(ROOT / "configs/techniques/catalog_v2.json")
    registry = AdaptiveTechniqueRegistry.from_catalog(catalog, project_root=ROOT)
    spec, candidate = load_adaptive_warm_start(
        WARM_START,
        project_root=ROOT,
        baseline=baseline,
        registry=registry,
    )
    return registry, baseline, spec, candidate


def test_historical_challenger_is_translated_onto_fixed_architecture() -> None:
    registry, baseline, spec, candidate = _loaded()
    inventory = registry.inventory()
    assert spec.source_candidate_id == "challenger-d4e040a07e6d"
    assert candidate.candidate_id.startswith("warm-start-")
    assert set(inventory.compulsory).issubset(candidate.techniques)
    assert set(candidate.techniques) - set(inventory.compulsory) == {
        "prior.quality",
        "ranking.top10_residual_reranker.v2",
    }
    assert "retrieval.sparse_only" not in candidate.techniques
    assert "residual_model_asset" not in spec.parameters
    assert registry.bindings["ranking.top10_residual_reranker.v2"].fit_required is True

    materialized = registry.materialize(baseline, candidate)
    assert materialized.architecture == "adaptive_hybrid_1a_3b_v1"
    assert materialized.extensions.quality_prior_weight == 0.2
    assert materialized.semantic_ranker.weight == baseline.semantic_ranker.weight
    assert materialized.union_ranker.backend == baseline.union_ranker.backend


def test_warm_seed_plans_drop_ones_and_complete_clean_control_add_ones() -> None:
    selected_config_bytes = (ROOT / CONFIG).read_bytes()
    registry, baseline, _, candidate = _loaded()
    engine = AdaptiveGhostLabEngine(
        baseline=baseline,
        registry=registry,
        warm_start=candidate,
        candidate_limit=36,
        beam_width=8,
    )
    plan = engine.initial_plan()
    coverage = engine.plan_coverage(plan.candidates)

    assert coverage["clean_control_verified"] is True
    assert set(engine.incumbent.techniques) == set(registry.inventory().compulsory)
    assert not (set(engine.incumbent.techniques) & set(registry.inventory().promotable))
    assert engine.baseline is baseline
    assert (ROOT / CONFIG).read_bytes() == selected_config_bytes
    assert coverage["add_one_coverage_complete"] is True
    assert set(coverage["add_one_covered"]) == set(registry.inventory().promotable)
    assert coverage["control_only_selected"] == ()
    assert set(coverage["control_only_explicitly_excluded"]) == set(
        registry.inventory().control_only
    )
    assert coverage["absorbed_control_coverage_complete"] is True
    assert coverage["missing_absorbed_control_parents"] == ()
    assert coverage["warm_drop_one_coverage_complete"] is True
    structural_count = sum(
        not item.candidate_id.startswith("semantic-calibration-")
        for item in plan.candidates
    )
    assert structural_count < engine.candidate_limit
    assert engine.candidate_limit - structural_count >= engine.limits.beam_width - 1

    rank_stack_add_ones = [
        item
        for item in plan.candidates
        if item.generation == "single" and "fusion.rank_stack.v1" in item.techniques
    ]
    assert len(rank_stack_add_ones) == 1
    assert set(rank_stack_add_ones[0].techniques) - set(
        registry.inventory().compulsory
    ) == {"fusion.rank_stack.v1", "ranking.fold_ensemble.v1"}

    ablations = {
        item.candidate_id: set(item.techniques) - set(registry.inventory().compulsory)
        for item in plan.candidates
        if item.candidate_id.startswith(f"{candidate.candidate_id}-without-")
    }
    assert ablations == {
        f"{candidate.candidate_id}-without-prior.quality": {
            "ranking.top10_residual_reranker.v2"
        },
        (f"{candidate.candidate_id}-without-ranking.top10_residual_reranker.v2"): {
            "prior.quality"
        },
    }


def test_warm_seed_is_evaluated_immediately_after_control() -> None:
    registry, baseline, _, candidate = _loaded()
    engine = AdaptiveGhostLabEngine(
        baseline=baseline,
        registry=registry,
        warm_start=candidate,
        candidate_limit=36,
        beam_width=8,
    )
    planned = engine.initial_plan().candidates
    assert candidate in planned
    assert planned[0] == engine.incumbent
    assert planned[1] == candidate
    warm_hash = engine.materialize(candidate).canonical_hash()
    assert (
        sum(engine.materialize(item).canonical_hash() == warm_hash for item in planned)
        == 1
    )

    seen: list[str] = []

    def evaluator(config, item, fidelity):  # type: ignore[no-untyped-def]
        seen.append(item.candidate_id)
        return AdaptiveEvaluation(
            candidate_id=item.candidate_id,
            fidelity=fidelity,
            score=0.8,
            session_rewards=(0.8, 0.8),
            hit_rate_at_10=0.9,
            mrr=0.7,
        )

    engine._evaluate_stage((planned[2], candidate, engine.incumbent), "f0", evaluator)
    assert seen[:2] == [engine.incumbent.candidate_id, candidate.candidate_id]

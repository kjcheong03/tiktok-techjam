from __future__ import annotations

import math

from ghostlab.campaign.catalog import TechniqueCatalog
from ghostlab.campaign.interaction_search import (
    CandidateEvidence,
    SearchLimits,
    plan_higher_order_round,
    plan_standalones_and_pairs,
)
from ghostlab.campaign.models import CandidateSpec, TechniqueSpec


def catalog() -> TechniqueCatalog:
    techniques = {
        "base": TechniqueSpec(id="base", family="baseline", availability="selected"),
        "a": TechniqueSpec(id="a", family="policy", availability="available"),
        "b": TechniqueSpec(id="b", family="ranking", availability="available"),
        "c": TechniqueSpec(id="c", family="retrieval", availability="available"),
        "d": TechniqueSpec(id="d", family="query", availability="available"),
        "needs-a": TechniqueSpec(
            id="needs-a",
            family="policy",
            availability="available",
            requires=("a",),
        ),
        "conflict": TechniqueSpec(
            id="conflict",
            family="ranking",
            availability="available",
            conflicts=("a",),
        ),
    }
    return TechniqueCatalog(2, techniques, "0" * 64)


def candidate(identifier: str, techniques: tuple[str, ...]) -> CandidateSpec:
    complexity = len(set(techniques) - {"base"})
    generation = "single" if complexity == 1 else "pair"
    return CandidateSpec(
        candidate_id=identifier,
        baseline_id="champion",
        techniques=techniques,
        complexity=complexity,
        generation=generation,
    )


def evidence(
    item: CandidateSpec,
    mean: float,
    lower: float,
    upper: float,
    *,
    repeats: int = 1,
    invalid: str | None = None,
    duplicate: str | None = None,
) -> CandidateEvidence:
    return CandidateEvidence(
        item.candidate_id,
        mean,
        lower,
        upper,
        repeated_evaluations=repeats,
        invalid_reason=invalid,
        duplicate_of=duplicate,
    )


def test_initial_plan_includes_all_standalones_then_caps_pairs() -> None:
    result = plan_standalones_and_pairs(
        catalog(),
        baseline_id="champion",
        baseline_techniques=("base",),
        technique_ids=("a", "b", "c", "d"),
        limits=SearchLimits(max_order=4, max_candidates=7),
    )
    assert sum(item.generation == "single" for item in result.candidates) == 4
    assert len(result.candidates) == 7
    assert result.cap_exhausted
    assert len({item.canonical_hash() for item in result.candidates}) == 7


def test_initial_plan_rejects_cap_that_cannot_fit_standalones() -> None:
    try:
        plan_standalones_and_pairs(
            catalog(),
            baseline_id="champion",
            baseline_techniques=("base",),
            technique_ids=("a", "b", "c", "d"),
            limits=SearchLimits(max_candidates=4),
        )
    except ValueError as error:
        assert "all executable standalones" in str(error)
    else:
        raise AssertionError("undersized cap was accepted")


def test_initial_plan_enumerates_every_compatible_pair_when_cap_allows() -> None:
    result = plan_standalones_and_pairs(
        catalog(),
        baseline_id="champion",
        baseline_techniques=("base",),
        technique_ids=("a", "b", "c", "d", "conflict"),
        limits=SearchLimits(max_order=4, max_candidates=20),
    )
    assert sum(item.generation == "single" for item in result.candidates) == 5
    assert sum(item.generation == "pair" for item in result.candidates) == 9
    assert not result.cap_exhausted


def test_pruning_retains_uncertain_and_mild_losers() -> None:
    positive = candidate("positive", ("base", "a"))
    uncertain = candidate("uncertain", ("base", "b"))
    mild = candidate("mild", ("base", "c"))
    strong_once = candidate("strong-once", ("base", "d"))
    dominated = candidate("dominated", ("base", "a", "b"))
    invalid = candidate("invalid", ("base", "a", "c"))
    duplicate = candidate("duplicate", ("base", "b", "c"))
    candidates = (
        positive,
        uncertain,
        mild,
        strong_once,
        dominated,
        invalid,
        duplicate,
    )
    records = {
        positive.candidate_id: evidence(positive, 0.02, 0.01, 0.03, repeats=2),
        uncertain.candidate_id: evidence(uncertain, -0.01, -0.03, 0.01, repeats=2),
        mild.candidate_id: evidence(mild, -0.004, -0.009, -0.001, repeats=2),
        strong_once.candidate_id: evidence(strong_once, -0.1, -0.2, -0.05, repeats=1),
        dominated.candidate_id: evidence(dominated, -0.03, -0.05, -0.01, repeats=2),
        invalid.candidate_id: evidence(
            invalid, 0.0, 0.0, 0.0, invalid="leakage detected"
        ),
        duplicate.candidate_id: evidence(
            duplicate, 0.0, 0.0, 0.0, duplicate="positive"
        ),
    }
    result = plan_higher_order_round(
        catalog(),
        evaluated_candidates=candidates,
        evidence=records,
        technique_ids=("a", "b", "c", "d", "needs-a"),
        baseline_techniques=("base",),
        limits=SearchLimits(max_order=4, max_candidates=20, beam_width=5),
        consumed_wall_seconds=0,
        estimated_candidate_seconds=1,
    )
    assert {"uncertain", "mild", "strong-once"} <= set(result.reserve_candidate_ids)
    assert {item.candidate_id for item in result.permanently_pruned} == {
        "dominated",
        "invalid",
        "duplicate",
    }
    assert {item.kind for item in result.permanently_pruned} == {
        "dominated",
        "invalid",
        "duplicate",
    }


def test_higher_order_search_is_deterministic_diverse_and_exploratory() -> None:
    parents = (
        candidate("policy", ("base", "a", "b")),
        candidate("retrieval", ("base", "b", "c")),
        candidate("uncertain", ("base", "c", "d")),
    )
    records = {
        "policy": evidence(parents[0], 0.03, 0.01, 0.05, repeats=2),
        "retrieval": evidence(parents[1], 0.02, 0.0, 0.04, repeats=2),
        "uncertain": evidence(parents[2], -0.002, -0.02, 0.01, repeats=2),
    }
    arguments = {
        "catalog": catalog(),
        "evaluated_candidates": parents,
        "evidence": records,
        "technique_ids": ("a", "b", "c", "d", "needs-a"),
        "baseline_techniques": ("base",),
        "limits": SearchLimits(
            max_order=4,
            max_candidates=9,
            beam_width=3,
            exploration_fraction=0.2,
            seed=11,
        ),
        "consumed_wall_seconds": 0,
        "estimated_candidate_seconds": 1,
    }
    first = plan_higher_order_round(**arguments)
    second = plan_higher_order_round(**arguments)
    assert first == second
    assert first.candidates
    assert len(first.candidates) <= 6
    assert first.exploration_candidate_ids
    assert len(first.exploration_candidate_ids) >= math.ceil(
        len(first.candidates) * 0.2
    )
    families = {
        catalog().techniques[item].family
        for result in first.candidates
        for item in result.techniques
        if item != "base"
    }
    assert {"policy", "ranking", "retrieval", "query"} <= families


def test_wall_and_candidate_budgets_stop_expansion() -> None:
    item = candidate("a", ("base", "a"))
    result = plan_higher_order_round(
        catalog(),
        evaluated_candidates=(item,),
        evidence={"a": evidence(item, 0.1, 0.05, 0.15, repeats=2)},
        technique_ids=("a", "b", "c"),
        baseline_techniques=("base",),
        limits=SearchLimits(max_candidates=10, max_wall_seconds=5),
        consumed_wall_seconds=5,
        estimated_candidate_seconds=1,
    )
    assert not result.candidates
    assert result.wall_exhausted


def test_pruning_audit_can_resurrect_contradicted_dominance() -> None:
    dominated = candidate("dominated", ("base", "a", "b"))
    limits = SearchLimits(
        max_candidates=10,
        pruning_audit_fraction=1.0,
        minimum_repeated_evidence=2,
    )
    original = evidence(dominated, -0.04, -0.06, -0.02, repeats=2)
    audit = evidence(dominated, 0.001, -0.01, 0.02, repeats=1)
    result = plan_higher_order_round(
        catalog(),
        evaluated_candidates=(dominated,),
        evidence={"dominated": original},
        technique_ids=("a", "b", "c"),
        baseline_techniques=("base",),
        limits=limits,
        consumed_wall_seconds=0,
        estimated_candidate_seconds=1,
        audit_evidence={"dominated": audit},
    )
    assert result.pruning_audit == (dominated,)
    assert result.resurrected == (dominated,)
    assert not result.permanently_pruned


def test_pending_pruning_audit_consumes_wall_budget_before_expansion() -> None:
    dominated = candidate("dominated", ("base", "a", "b"))
    result = plan_higher_order_round(
        catalog(),
        evaluated_candidates=(dominated,),
        evidence={"dominated": evidence(dominated, -0.04, -0.06, -0.02, repeats=2)},
        technique_ids=("a", "b", "c"),
        baseline_techniques=("base",),
        limits=SearchLimits(
            max_candidates=10,
            max_wall_seconds=1,
            pruning_audit_fraction=1.0,
        ),
        consumed_wall_seconds=0,
        estimated_candidate_seconds=1,
    )
    assert result.pruning_audit == (dominated,)
    assert not result.candidates
    assert result.wall_exhausted


def test_dependencies_and_conflicts_are_validated_during_expansion() -> None:
    parent = candidate("parent", ("base", "b"))
    result = plan_higher_order_round(
        catalog(),
        evaluated_candidates=(parent,),
        evidence={"parent": evidence(parent, 0.1, 0.05, 0.2, repeats=2)},
        technique_ids=("a", "b", "needs-a", "conflict"),
        baseline_techniques=("base",),
        limits=SearchLimits(max_order=4, max_candidates=10),
        consumed_wall_seconds=0,
        estimated_candidate_seconds=1,
    )
    assert any({"a", "needs-a"} <= set(item.techniques) for item in result.candidates)
    assert all(
        not ({"a", "conflict"} <= set(item.techniques)) for item in result.candidates
    )


def test_expansion_deduplicates_structure_across_generation_provenance() -> None:
    parent = candidate("parent", ("base", "a", "b"))
    already_evaluated = CandidateSpec(
        candidate_id="existing-triple",
        baseline_id="champion",
        techniques=("base", "a", "b", "c"),
        complexity=3,
        generation="triple",
    )
    result = plan_higher_order_round(
        catalog(),
        evaluated_candidates=(parent, already_evaluated),
        evidence={
            "parent": evidence(parent, 0.1, 0.05, 0.2, repeats=2),
            "existing-triple": evidence(already_evaluated, 0.1, 0.05, 0.2, repeats=2),
        },
        technique_ids=("a", "b", "c"),
        baseline_techniques=("base",),
        limits=SearchLimits(max_order=4, max_candidates=10),
        consumed_wall_seconds=0,
        estimated_candidate_seconds=1,
    )
    assert all(
        set(item.techniques) != {"base", "a", "b", "c"} for item in result.candidates
    )

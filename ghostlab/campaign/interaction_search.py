from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

from ghostlab.campaign.catalog import TechniqueCatalog
from ghostlab.campaign.compatibility import validate_techniques
from ghostlab.campaign.models import CandidateSpec
from ghostlab.campaign.planner import SkippedCandidate, plan_candidates


@dataclass(frozen=True)
class SearchLimits:
    """Prospective bounds for one structure-search round."""

    max_order: int = 4
    max_candidates: int = 500
    max_wall_seconds: float = 36 * 60 * 60
    beam_width: int = 24
    exploration_fraction: float = 0.2
    pruning_audit_fraction: float = 0.1
    minimum_repeated_evidence: int = 2
    mild_loss_floor: float = -0.005
    domination_margin: float = 0.0
    seed: int = 20260826

    def __post_init__(self) -> None:
        if not 1 <= self.max_order <= 64:
            raise ValueError("max_order must be between one and 64")
        if self.max_candidates <= 0 or self.beam_width <= 0:
            raise ValueError("candidate and beam bounds must be positive")
        if self.max_wall_seconds <= 0:
            raise ValueError("wall budget must be positive")
        if not 0.0 <= self.exploration_fraction <= 0.5:
            raise ValueError("exploration_fraction must be in [0, 0.5]")
        if not 0.0 <= self.pruning_audit_fraction <= 1.0:
            raise ValueError("pruning_audit_fraction must be in [0, 1]")
        if self.minimum_repeated_evidence < 2:
            raise ValueError("permanent domination pruning requires repeated evidence")
        if self.mild_loss_floor > 0.0 or self.domination_margin < 0.0:
            raise ValueError("invalid pruning thresholds")


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    mean_delta: float
    confidence_lower: float
    confidence_upper: float
    repeated_evaluations: int = 1
    invalid_reason: str | None = None
    duplicate_of: str | None = None

    def __post_init__(self) -> None:
        if self.confidence_lower > self.confidence_upper:
            raise ValueError("confidence interval is reversed")
        if self.repeated_evaluations <= 0:
            raise ValueError("repeated_evaluations must be positive")


PruneKind = Literal["invalid", "duplicate", "dominated"]


@dataclass(frozen=True)
class PruneDecision:
    candidate_id: str
    kind: PruneKind
    reason: str
    repeated_evaluations: int


@dataclass(frozen=True)
class InteractionSearchPlan:
    candidates: tuple[CandidateSpec, ...]
    skipped: tuple[SkippedCandidate, ...]
    reserve_candidate_ids: tuple[str, ...] = ()
    exploration_candidate_ids: tuple[str, ...] = ()
    permanently_pruned: tuple[PruneDecision, ...] = ()
    pruning_audit: tuple[CandidateSpec, ...] = ()
    resurrected: tuple[CandidateSpec, ...] = ()
    cap_exhausted: bool = False
    wall_exhausted: bool = False


def plan_standalones_and_pairs(
    catalog: TechniqueCatalog,
    *,
    baseline_id: str,
    baseline_techniques: tuple[str, ...],
    technique_ids: tuple[str, ...],
    limits: SearchLimits,
) -> InteractionSearchPlan:
    """Plan every compatible standalone, then pairs until the hard cap."""

    requested = tuple(sorted(set(technique_ids) - set(baseline_techniques)))
    standalone_plan = plan_candidates(
        catalog,
        baseline_id=baseline_id,
        baseline_techniques=baseline_techniques,
        technique_ids=requested,
        max_order=1,
        candidate_limit=max(1, len(requested) + 1),
    )
    minimum_cap = len(standalone_plan.candidates)
    if limits.max_candidates < minimum_cap:
        raise ValueError(
            "max_candidates cannot fit the control and all executable standalones"
        )
    plan = plan_candidates(
        catalog,
        baseline_id=baseline_id,
        baseline_techniques=baseline_techniques,
        technique_ids=requested,
        max_order=min(2, limits.max_order),
        candidate_limit=limits.max_candidates,
    )
    return InteractionSearchPlan(
        candidates=plan.candidates,
        skipped=plan.skipped,
        cap_exhausted=len(plan.candidates) >= limits.max_candidates,
    )


def _structure_hash(candidate: CandidateSpec) -> str:
    """Hash behaviorally relevant structure, independent of search provenance."""

    payload = {
        "baseline_id": candidate.baseline_id,
        "techniques": sorted(candidate.techniques),
        "parameters": sorted(candidate.parameters, key=lambda item: repr(item)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _permanent_prune(
    candidate: CandidateSpec,
    evidence: CandidateEvidence | None,
    limits: SearchLimits,
) -> PruneDecision | None:
    if evidence is None:
        return None
    if evidence.invalid_reason is not None:
        return PruneDecision(
            candidate.candidate_id,
            "invalid",
            evidence.invalid_reason,
            evidence.repeated_evaluations,
        )
    if evidence.duplicate_of is not None:
        return PruneDecision(
            candidate.candidate_id,
            "duplicate",
            f"behaviorally duplicates {evidence.duplicate_of}",
            evidence.repeated_evaluations,
        )
    repeatedly_dominated = (
        evidence.repeated_evaluations >= limits.minimum_repeated_evidence
        and evidence.mean_delta < limits.mild_loss_floor
        and evidence.confidence_upper < -limits.domination_margin
        and candidate.complexity > 0
    )
    if repeatedly_dominated:
        return PruneDecision(
            candidate.candidate_id,
            "dominated",
            "upper confidence bound remains below the matched control after "
            "declared repeated evaluations",
            evidence.repeated_evaluations,
        )
    return None


def _is_reserve(evidence: CandidateEvidence | None, limits: SearchLimits) -> bool:
    if evidence is None:
        return True
    uncertain = evidence.confidence_lower <= 0.0 <= evidence.confidence_upper
    mild_loser = evidence.mean_delta >= limits.mild_loss_floor
    not_repeated = evidence.repeated_evaluations < limits.minimum_repeated_evidence
    return uncertain or mild_loser or not_repeated


def _family_signature(
    catalog: TechniqueCatalog,
    candidate: CandidateSpec,
    baseline: set[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                catalog.techniques[item].family
                for item in candidate.techniques
                if item not in baseline and item in catalog.techniques
            }
        )
    ) or ("control",)


def _evidence_order(
    candidate: CandidateSpec, evidence: CandidateEvidence | None
) -> tuple[float, float, int, str]:
    if evidence is None:
        return (
            float("inf"),
            float("inf"),
            candidate.complexity,
            candidate.candidate_id,
        )
    return (
        -evidence.confidence_upper,
        -evidence.mean_delta,
        candidate.complexity,
        candidate.candidate_id,
    )


def _diverse_beam(
    catalog: TechniqueCatalog,
    candidates: tuple[CandidateSpec, ...],
    evidence: dict[str, CandidateEvidence],
    *,
    baseline: set[str],
    width: int,
) -> tuple[CandidateSpec, ...]:
    groups: dict[tuple[str, ...], list[CandidateSpec]] = {}
    for candidate in candidates:
        groups.setdefault(_family_signature(catalog, candidate, baseline), []).append(
            candidate
        )
    for members in groups.values():
        members.sort(
            key=lambda item: _evidence_order(item, evidence.get(item.candidate_id))
        )
    selected: list[CandidateSpec] = []
    signatures = sorted(groups)
    while len(selected) < width and any(groups.values()):
        for signature in signatures:
            if groups[signature] and len(selected) < width:
                selected.append(groups[signature].pop(0))
    return tuple(selected)


def _deterministic_exploration_order(
    candidates: tuple[CandidateSpec, ...], seed: int
) -> tuple[CandidateSpec, ...]:
    def key(candidate: CandidateSpec) -> tuple[str, str]:
        encoded = f"{seed}\0{_structure_hash(candidate)}".encode()
        return hashlib.sha256(encoded).hexdigest(), candidate.candidate_id

    return tuple(sorted(candidates, key=key))


def _diverse_exploration_beam(
    catalog: TechniqueCatalog,
    candidates: tuple[CandidateSpec, ...],
    *,
    baseline: set[str],
    width: int,
    seed: int,
) -> tuple[CandidateSpec, ...]:
    """Take a seeded round-robin sample across technique-family signatures."""

    groups: dict[tuple[str, ...], list[CandidateSpec]] = {}
    for candidate in _deterministic_exploration_order(candidates, seed):
        groups.setdefault(_family_signature(catalog, candidate, baseline), []).append(
            candidate
        )
    selected: list[CandidateSpec] = []
    signatures = sorted(groups)
    while len(selected) < width and any(groups.values()):
        for signature in signatures:
            if groups[signature] and len(selected) < width:
                selected.append(groups[signature].pop(0))
    return tuple(selected)


def _dependency_closure(
    catalog: TechniqueCatalog, selected: set[str]
) -> tuple[str, ...]:
    pending = list(selected)
    while pending:
        technique = catalog.techniques.get(pending.pop())
        if technique is None:
            continue
        for required in technique.requires:
            if required not in selected:
                selected.add(required)
                pending.append(required)
    return tuple(sorted(selected))


def _expanded_candidate(
    catalog: TechniqueCatalog,
    parent: CandidateSpec,
    technique_id: str,
    baseline: set[str],
) -> CandidateSpec | None:
    complete = _dependency_closure(catalog, set(parent.techniques) | {technique_id})
    if not validate_techniques(catalog, complete).valid:
        return None
    complexity = len(set(complete) - baseline)
    generation: Literal["triple", "beam"] = "triple" if complexity == 3 else "beam"
    candidate = CandidateSpec(
        candidate_id="pending",
        baseline_id=parent.baseline_id,
        techniques=complete,
        parameters=parent.parameters,
        complexity=complexity,
        generation=generation,
    )
    return candidate.model_copy(
        update={"candidate_id": f"challenger-{_structure_hash(candidate)[:12]}"}
    )


def _audit_sample(
    candidates: tuple[CandidateSpec, ...],
    pruned: tuple[PruneDecision, ...],
    limits: SearchLimits,
) -> tuple[CandidateSpec, ...]:
    by_id = {item.candidate_id: item for item in candidates}
    eligible = tuple(
        by_id[item.candidate_id]
        for item in pruned
        if item.kind == "dominated" and item.candidate_id in by_id
    )
    if not eligible or limits.pruning_audit_fraction == 0.0:
        return ()
    count = max(1, math.ceil(len(eligible) * limits.pruning_audit_fraction))
    return _deterministic_exploration_order(eligible, limits.seed)[:count]


def _resurrections(
    audit: tuple[CandidateSpec, ...],
    audit_evidence: dict[str, CandidateEvidence],
    limits: SearchLimits,
) -> tuple[CandidateSpec, ...]:
    resurrected = []
    for candidate in audit:
        evidence = audit_evidence.get(candidate.candidate_id)
        if evidence is None or evidence.invalid_reason or evidence.duplicate_of:
            continue
        no_longer_dominated = evidence.confidence_upper >= -limits.domination_margin
        if no_longer_dominated:
            resurrected.append(candidate)
    return tuple(sorted(resurrected, key=lambda item: item.candidate_id))


def plan_higher_order_round(
    catalog: TechniqueCatalog,
    *,
    evaluated_candidates: tuple[CandidateSpec, ...],
    evidence: dict[str, CandidateEvidence],
    technique_ids: tuple[str, ...],
    baseline_techniques: tuple[str, ...],
    limits: SearchLimits,
    consumed_wall_seconds: float,
    estimated_candidate_seconds: float,
    audit_evidence: dict[str, CandidateEvidence] | None = None,
) -> InteractionSearchPlan:
    """Prune conservatively, then expand a diverse beam with exploration reserve."""

    if consumed_wall_seconds < 0.0 or estimated_candidate_seconds <= 0.0:
        raise ValueError("invalid wall-budget inputs")
    baseline = set(baseline_techniques)
    historical_pruned = tuple(
        decision
        for candidate in evaluated_candidates
        if (
            decision := _permanent_prune(
                candidate, evidence.get(candidate.candidate_id), limits
            )
        )
        is not None
    )
    remaining_wall = max(0.0, limits.max_wall_seconds - consumed_wall_seconds)
    wall_slots = int(remaining_wall // estimated_candidate_seconds)
    requested_audit = _audit_sample(evaluated_candidates, historical_pruned, limits)
    completed_audit_ids = set(audit_evidence or {})
    pending_audit = tuple(
        item for item in requested_audit if item.candidate_id not in completed_audit_ids
    )[:wall_slots]
    pending_audit_ids = {item.candidate_id for item in pending_audit}
    audit = tuple(
        item
        for item in requested_audit
        if item.candidate_id in completed_audit_ids
        or item.candidate_id in pending_audit_ids
    )
    wall_slots -= len(pending_audit)
    resurrected = _resurrections(audit, audit_evidence or {}, limits)
    resurrected_ids = {item.candidate_id for item in resurrected}
    pruned = tuple(
        item for item in historical_pruned if item.candidate_id not in resurrected_ids
    )
    pruned_ids = {item.candidate_id for item in pruned}
    survivors = tuple(
        item for item in evaluated_candidates if item.candidate_id not in pruned_ids
    )
    survivor_by_hash = {
        _structure_hash(item): item for item in (*survivors, *resurrected)
    }
    survivors = tuple(
        sorted(survivor_by_hash.values(), key=lambda item: item.candidate_id)
    )
    effective_evidence = dict(evidence)
    effective_evidence.update(audit_evidence or {})
    reserve_ids = tuple(
        sorted(
            item.candidate_id
            for item in survivors
            if _is_reserve(effective_evidence.get(item.candidate_id), limits)
        )
    )

    candidate_slots = max(0, limits.max_candidates - len(evaluated_candidates))
    available_slots = min(wall_slots, candidate_slots)
    if available_slots == 0 or limits.max_order <= 2:
        return InteractionSearchPlan(
            candidates=(),
            skipped=(),
            reserve_candidate_ids=reserve_ids,
            permanently_pruned=pruned,
            pruning_audit=audit,
            resurrected=resurrected,
            cap_exhausted=candidate_slots == 0,
            wall_exhausted=wall_slots == 0,
        )

    exploration_width = (
        max(1, math.ceil(limits.beam_width * limits.exploration_fraction))
        if limits.exploration_fraction > 0.0
        else 0
    )
    exploitation_width = max(1, limits.beam_width - exploration_width)
    exploitation_parents = _diverse_beam(
        catalog,
        survivors,
        effective_evidence,
        baseline=baseline,
        width=min(exploitation_width, len(survivors)),
    )
    reserve_id_set = set(reserve_ids)
    exploration_pool = tuple(
        item for item in survivors if item.candidate_id in reserve_id_set
    )
    exploration_parents = _diverse_exploration_beam(
        catalog,
        exploration_pool,
        baseline=baseline,
        width=min(exploration_width, len(exploration_pool)),
        seed=limits.seed,
    )

    roots = tuple(
        sorted(
            item
            for item in set(technique_ids) - baseline
            if item in catalog.techniques and catalog.techniques[item].executable
        )
    )
    existing_hashes = {_structure_hash(item) for item in evaluated_candidates}

    def expansions(parents: tuple[CandidateSpec, ...]) -> tuple[CandidateSpec, ...]:
        produced: dict[str, CandidateSpec] = {}
        for parent in parents:
            for technique_id in roots:
                if technique_id in parent.techniques:
                    continue
                candidate = _expanded_candidate(catalog, parent, technique_id, baseline)
                if candidate is None or candidate.complexity > limits.max_order:
                    continue
                key = _structure_hash(candidate)
                if key not in existing_hashes:
                    produced.setdefault(key, candidate)
        return tuple(sorted(produced.values(), key=lambda item: item.candidate_id))

    exploitation_pool = expansions(exploitation_parents)
    exploration_expansions = _deterministic_exploration_order(
        expansions(exploration_parents), limits.seed
    )
    exploration_slots = (
        math.ceil(available_slots * limits.exploration_fraction)
        if exploration_expansions and limits.exploration_fraction > 0.0
        else 0
    )
    exploration_selected = exploration_expansions[:exploration_slots]
    selected_hashes = {_structure_hash(item) for item in exploration_selected}
    exploitation_options = tuple(
        item
        for item in exploitation_pool
        if _structure_hash(item) not in selected_hashes
    )
    exploitation_selected = _diverse_beam(
        catalog,
        exploitation_options,
        {},
        baseline=baseline,
        width=available_slots - len(exploration_selected),
    )
    selected_hashes.update(_structure_hash(item) for item in exploitation_selected)
    remaining_slots = (
        available_slots - len(exploration_selected) - len(exploitation_selected)
    )
    exploration_fill = tuple(
        item
        for item in exploration_expansions[exploration_slots:]
        if _structure_hash(item) not in selected_hashes
    )[:remaining_slots]
    exploration_selected = (*exploration_selected, *exploration_fill)
    selected = tuple(
        sorted(
            (*exploitation_selected, *exploration_selected),
            key=lambda item: item.candidate_id,
        )
    )
    return InteractionSearchPlan(
        candidates=selected,
        skipped=(),
        reserve_candidate_ids=reserve_ids,
        exploration_candidate_ids=tuple(
            sorted(item.candidate_id for item in exploration_selected)
        ),
        permanently_pruned=pruned,
        pruning_audit=audit,
        resurrected=resurrected,
        cap_exhausted=len(selected) >= candidate_slots,
        wall_exhausted=len(selected) >= wall_slots,
    )

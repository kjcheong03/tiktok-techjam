from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations

from ghostlab.campaign.catalog import TechniqueCatalog
from ghostlab.campaign.compatibility import validate_techniques
from ghostlab.campaign.models import CandidateSpec


@dataclass(frozen=True)
class SkippedCandidate:
    roots: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidatePlan:
    candidates: tuple[CandidateSpec, ...]
    skipped: tuple[SkippedCandidate, ...]


def _dependency_closure(
    catalog: TechniqueCatalog, selected: set[str]
) -> tuple[str, ...]:
    pending = list(selected)
    while pending:
        technique_id = pending.pop()
        technique = catalog.techniques.get(technique_id)
        if technique is None:
            continue
        for required in technique.requires:
            if required not in selected:
                selected.add(required)
                pending.append(required)
    return tuple(sorted(selected))


def _candidate_id(baseline_id: str, roots: tuple[str, ...]) -> str:
    encoded = "\0".join((baseline_id, *sorted(roots))).encode()
    return f"challenger-{hashlib.sha256(encoded).hexdigest()[:12]}"


def plan_candidates(
    catalog: TechniqueCatalog,
    *,
    baseline_id: str,
    baseline_techniques: tuple[str, ...],
    technique_ids: tuple[str, ...],
    max_order: int,
    candidate_limit: int = 1000,
) -> CandidatePlan:
    if max_order <= 0:
        raise ValueError("max_order must be positive")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    unknown_baseline = set(baseline_techniques) - set(catalog.techniques)
    if unknown_baseline:
        raise ValueError(f"unknown baseline techniques: {sorted(unknown_baseline)}")
    candidates: list[CandidateSpec] = [
        CandidateSpec(
            candidate_id=f"control-{baseline_id}",
            baseline_id=baseline_id,
            techniques=tuple(sorted(set(baseline_techniques))),
            complexity=0,
            generation="control",
        )
    ]
    skipped: list[SkippedCandidate] = []
    seen = {candidates[0].canonical_hash()}
    roots = tuple(sorted(set(technique_ids) - set(baseline_techniques)))
    for order in range(1, min(max_order, len(roots)) + 1):
        generation = {1: "single", 2: "pair", 3: "triple"}.get(order, "beam")
        for additions in combinations(roots, order):
            if len(candidates) >= candidate_limit:
                return CandidatePlan(tuple(candidates), tuple(skipped))
            complete = _dependency_closure(
                catalog, set(baseline_techniques) | set(additions)
            )
            result = validate_techniques(catalog, complete)
            if not result.valid:
                skipped.append(SkippedCandidate(additions, result.reasons))
                continue
            candidate = CandidateSpec(
                candidate_id=_candidate_id(baseline_id, additions),
                baseline_id=baseline_id,
                techniques=complete,
                complexity=len(set(complete) - set(baseline_techniques)),
                generation=generation,  # type: ignore[arg-type]
            )
            key = candidate.canonical_hash()
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return CandidatePlan(tuple(candidates), tuple(skipped))


def backward_ablations(candidate: CandidateSpec) -> tuple[CandidateSpec, ...]:
    if candidate.generation == "control" or not candidate.techniques:
        return ()
    results: list[CandidateSpec] = []
    for removed in candidate.techniques:
        techniques = tuple(item for item in candidate.techniques if item != removed)
        results.append(
            CandidateSpec(
                candidate_id=f"{candidate.candidate_id}-without-{removed}",
                baseline_id=candidate.baseline_id,
                techniques=techniques,
                parameters=candidate.parameters,
                complexity=max(0, candidate.complexity - 1),
                generation="ablation",
            )
        )
    return tuple(results)

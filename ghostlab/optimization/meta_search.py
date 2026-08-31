from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Literal

MetaStrategy = Literal["random", "grid", "beam", "allocated"]


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    family: str
    complexity: int
    f0_score: float
    f2_score: float


@dataclass(frozen=True)
class MetaSearchResult:
    strategy: MetaStrategy
    selected_id: str
    selected_score: float
    best_observed_id: str
    best_observed_score: float
    session_evaluations: int
    screened: int
    promoted: int


def _tie_band_select(
    candidates: list[CandidateEvidence], tie_band: float
) -> CandidateEvidence:
    best = max(candidate.f2_score for candidate in candidates)
    eligible = [
        candidate for candidate in candidates if candidate.f2_score >= best - tie_band
    ]
    return min(
        eligible,
        key=lambda candidate: (
            candidate.complexity,
            -candidate.f2_score,
            candidate.candidate_id,
        ),
    )


def _allocated_screen(
    pool: list[CandidateEvidence], count: int, rng: random.Random, exploration: float
) -> list[CandidateEvidence]:
    by_family: dict[str, list[CandidateEvidence]] = {}
    for candidate in pool:
        by_family.setdefault(candidate.family, []).append(candidate)
    for candidates in by_family.values():
        rng.shuffle(candidates)

    observations: dict[str, list[float]] = {family: [] for family in by_family}
    selected: list[CandidateEvidence] = []
    while len(selected) < count:
        available = [family for family, values in by_family.items() if values]
        if not available:
            break
        unseen = [family for family in available if not observations[family]]
        if unseen:
            family = rng.choice(sorted(unseen))
        else:
            trials = len(selected)
            family = max(
                available,
                key=lambda name: (
                    statistics.fmean(observations[name])
                    + exploration
                    * math.sqrt(math.log(trials + 1) / len(observations[name])),
                    name,
                ),
            )
        candidate = by_family[family].pop()
        selected.append(candidate)
        observations[family].append(candidate.f0_score)
    return selected


def cost_aware_search(
    candidates: list[CandidateEvidence],
    *,
    strategy: MetaStrategy,
    budget: int,
    f0_cost: int,
    f2_cost: int,
    seed: int,
    tie_band: float = 0.01,
    screen_fraction: float = 0.5,
    exploration: float = 0.2,
) -> MetaSearchResult:
    """Replay a search strategy under a session-evaluation budget.

    F0 observations are reused during F2 promotion, so a promoted candidate costs
    ``f2_cost - f0_cost`` additional sessions rather than another full F2 pass.
    """

    if not candidates:
        raise ValueError("candidate pool cannot be empty")
    if f0_cost <= 0 or f2_cost <= f0_cost:
        raise ValueError("costs must satisfy 0 < f0_cost < f2_cost")
    if budget < f2_cost:
        raise ValueError("budget must fund at least one F2 evaluation")
    if not 0.0 < screen_fraction < 1.0:
        raise ValueError("screen_fraction must be between zero and one")

    rng = random.Random(seed)
    pool = list(candidates)
    if strategy == "grid":
        ordered = sorted(pool, key=lambda candidate: candidate.candidate_id)
    else:
        ordered = list(pool)
        rng.shuffle(ordered)

    if strategy in {"random", "grid"}:
        promoted = ordered[: min(len(ordered), budget // f2_cost)]
        used = len(promoted) * f2_cost
        screened_count = 0
    else:
        screen_count = min(len(pool), int(budget * screen_fraction) // f0_cost)
        if strategy == "allocated":
            screened = _allocated_screen(pool, screen_count, rng, exploration)
        else:
            screened = ordered[:screen_count]
        remaining = budget - len(screened) * f0_cost
        promotion_count = min(len(screened), remaining // (f2_cost - f0_cost))
        promoted = sorted(
            screened,
            key=lambda candidate: (
                -candidate.f0_score,
                candidate.complexity,
                candidate.candidate_id,
            ),
        )[:promotion_count]
        used = len(screened) * f0_cost + len(promoted) * (f2_cost - f0_cost)
        screened_count = len(screened)

    if not promoted:
        raise ValueError("budget allocation produced no F2 promotions")
    best = min(
        promoted,
        key=lambda candidate: (-candidate.f2_score, candidate.candidate_id),
    )
    selected = _tie_band_select(promoted, tie_band)
    return MetaSearchResult(
        strategy=strategy,
        selected_id=selected.candidate_id,
        selected_score=selected.f2_score,
        best_observed_id=best.candidate_id,
        best_observed_score=best.f2_score,
        session_evaluations=used,
        screened=screened_count,
        promoted=len(promoted),
    )

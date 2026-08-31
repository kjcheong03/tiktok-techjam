from __future__ import annotations

import statistics
from dataclasses import dataclass

from ghostlab.evaluation.statistics import (
    bootstrap_mean_interval,
    paired_randomization_pvalue,
)
from ghostlab.optimization.search import interaction_gain


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    complexity: int
    score: float
    session_rewards: tuple[float, ...]
    scenario_scores: dict[str, float]
    latency_p95_ms: float = 0.0
    memory_mb: float = 0.0
    hit_rate_at_10: float | None = None
    mrr: float | None = None
    mttc: float | None = None


@dataclass(frozen=True)
class PairedAnalysis:
    candidate_id: str
    baseline_id: str
    mean_delta: float
    confidence_interval: tuple[float, float]
    randomization_pvalue: float
    wins: int
    ties: int
    losses: int
    scenario_deltas: dict[str, float]


def paired_analysis(
    candidate: CandidateEvaluation,
    baseline: CandidateEvaluation,
    *,
    resamples: int = 5000,
    seed: int = 20260826,
) -> PairedAnalysis:
    if len(candidate.session_rewards) != len(baseline.session_rewards):
        raise ValueError("paired evaluations must contain the same number of sessions")
    if not candidate.session_rewards:
        raise ValueError("paired evaluations cannot be empty")
    deltas = tuple(
        left - right
        for left, right in zip(
            candidate.session_rewards, baseline.session_rewards, strict=True
        )
    )
    scenarios = set(candidate.scenario_scores) | set(baseline.scenario_scores)
    return PairedAnalysis(
        candidate_id=candidate.candidate_id,
        baseline_id=baseline.candidate_id,
        mean_delta=statistics.fmean(deltas),
        confidence_interval=bootstrap_mean_interval(
            deltas, resamples=resamples, seed=seed
        ),
        randomization_pvalue=paired_randomization_pvalue(
            deltas, resamples=resamples, seed=seed
        ),
        wins=sum(value > 0.0 for value in deltas),
        ties=sum(value == 0.0 for value in deltas),
        losses=sum(value < 0.0 for value in deltas),
        scenario_deltas={
            scenario: candidate.scenario_scores.get(scenario, 0.0)
            - baseline.scenario_scores.get(scenario, 0.0)
            for scenario in sorted(scenarios)
        },
    )


def interaction_analysis(
    base: CandidateEvaluation,
    first: CandidateEvaluation,
    second: CandidateEvaluation,
    combination: CandidateEvaluation,
) -> tuple[float, tuple[float, ...]]:
    lengths = {len(item.session_rewards) for item in (base, first, second, combination)}
    if len(lengths) != 1 or not base.session_rewards:
        raise ValueError("interaction evaluations require aligned non-empty sessions")
    session_interactions = tuple(
        both - left - right + control
        for control, left, right, both in zip(
            base.session_rewards,
            first.session_rewards,
            second.session_rewards,
            combination.session_rewards,
            strict=True,
        )
    )
    aggregate = interaction_gain(
        base.score, first.score, second.score, combination.score
    )
    return aggregate, session_interactions

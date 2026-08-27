from __future__ import annotations

from dataclasses import dataclass

from ghostlab.campaign.analyze import CandidateEvaluation, PairedAnalysis


@dataclass(frozen=True)
class CandidateProposal:
    candidate_id: str
    score: float
    mean_delta: float
    complexity: int
    reason: str


def propose_candidate(
    evaluations: tuple[CandidateEvaluation, ...],
    analyses: dict[str, PairedAnalysis],
    *,
    tie_band: float = 0.005,
    maximum_scenario_regression: float = 0.02,
) -> CandidateProposal:
    if not evaluations:
        raise ValueError("proposal requires candidate evaluations")
    safe = [
        item
        for item in evaluations
        if item.candidate_id in analyses
        and all(
            delta >= -maximum_scenario_regression
            for delta in analyses[item.candidate_id].scenario_deltas.values()
        )
    ]
    if not safe:
        raise ValueError("no candidate passed the declared scenario gate")
    best_score = max(item.score for item in safe)
    eligible = [item for item in safe if item.score >= best_score - tie_band]
    selected = min(
        eligible,
        key=lambda item: (
            item.complexity,
            item.latency_p95_ms,
            item.memory_mb,
            -item.score,
            item.candidate_id,
        ),
    )
    analysis = analyses[selected.candidate_id]
    return CandidateProposal(
        candidate_id=selected.candidate_id,
        score=selected.score,
        mean_delta=analysis.mean_delta,
        complexity=selected.complexity,
        reason=(
            "proposal only: passed scenario gate and selected inside the declared "
            "score tie band using complexity/resource tie-breaks"
        ),
    )

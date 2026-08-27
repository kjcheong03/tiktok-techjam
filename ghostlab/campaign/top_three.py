from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Literal

from ghostlab.campaign.analyze import CandidateEvaluation, PairedAnalysis
from ghostlab.research.technique_suite import UnifiedTechniqueConfig

ProposalRole = Literal["score_leader", "robust_leader", "efficient_alternative"]
ALLOWED_EXTRAS = frozenset({"core", "gbdt", "dense", "neural", "all"})
CONFIG_PATH_FIELDS = (
    "compiled_config_path",
    "learned_question_asset",
    "joint_policy_asset",
    "normalizer_asset",
    "dense_model_path",
    "reranker_model_asset",
    "cross_encoder_model_path",
    "learned_sparse_asset",
    "late_interaction_asset",
    "router_asset",
)


@dataclass(frozen=True)
class CandidatePackage:
    candidate_id: str
    config: UnifiedTechniqueConfig
    dependency_extras: tuple[str, ...] = ("core",)
    assets: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    confirmed: bool = False
    safe: bool = False
    notes: tuple[str, ...] = ()
    enabled_techniques: tuple[str, ...] = ()
    technique_sources: tuple[tuple[str, str, str], ...] = ()
    tuned_parameters: tuple[tuple[str, str | int | float | bool], ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate package requires an ID")
        unknown = set(self.dependency_extras) - ALLOWED_EXTRAS
        if unknown:
            raise ValueError(f"unknown dependency extras: {sorted(unknown)}")
        if len(set(self.dependency_extras)) != len(self.dependency_extras):
            raise ValueError("dependency extras must be unique")
        for value in (*self.assets, *self.evidence_refs):
            _safe_project_path(value)
        configured_paths = {
            value
            for field in CONFIG_PATH_FIELDS
            for value in (getattr(self.config, field),)
            if value is not None
        }
        missing = configured_paths - set(self.assets)
        if missing:
            raise ValueError(
                f"candidate package must declare every config asset: {sorted(missing)}"
            )


@dataclass(frozen=True)
class ProposalCandidate:
    role: ProposalRole
    evaluation: CandidateEvaluation
    analysis: PairedAnalysis
    package: CandidatePackage
    reason: str


@dataclass(frozen=True)
class TopThreeSelection:
    baseline_id: str
    score_leader: ProposalCandidate
    robust_leader: ProposalCandidate
    efficient_alternative: ProposalCandidate
    excluded: tuple[tuple[str, str], ...]

    @property
    def candidates(self) -> tuple[ProposalCandidate, ...]:
        return (
            self.score_leader,
            self.robust_leader,
            self.efficient_alternative,
        )


def select_top_three(
    evaluations: tuple[CandidateEvaluation, ...],
    analyses: dict[str, PairedAnalysis],
    packages: dict[str, CandidatePackage],
    *,
    baseline_id: str,
    project_root: str | Path,
    maximum_candidates: int = 1000,
    maximum_scenario_regression: float = 0.02,
    minimum_mean_delta: float = 0.0,
    maximum_latency_p95_ms: float = 1000.0,
    maximum_memory_mb: float = 4096.0,
    efficient_score_band: float = 0.02,
) -> TopThreeSelection:
    """Return three distinct human-review proposals from validated OOF evidence.

    This function cannot promote, write presets, or inspect protected data. Package
    readiness is explicit, and every referenced asset/evidence path must already
    exist below ``project_root``.
    """
    if not baseline_id:
        raise ValueError("baseline ID is required")
    if not 3 <= maximum_candidates <= 10000:
        raise ValueError("maximum_candidates must be between 3 and 10000")
    if len(evaluations) > maximum_candidates:
        raise ValueError("candidate evaluations exceed the declared bound")
    if len({item.candidate_id for item in evaluations}) != len(evaluations):
        raise ValueError("candidate evaluation IDs must be unique")
    if maximum_scenario_regression < 0.0 or efficient_score_band < 0.0:
        raise ValueError("selection tolerances cannot be negative")
    root = Path(project_root).resolve()
    safe: list[tuple[CandidateEvaluation, PairedAnalysis, CandidatePackage]] = []
    excluded: list[tuple[str, str]] = []
    for evaluation in sorted(evaluations, key=lambda item: item.candidate_id):
        reason = _exclusion_reason(
            evaluation,
            analyses.get(evaluation.candidate_id),
            packages.get(evaluation.candidate_id),
            baseline_id=baseline_id,
            root=root,
            maximum_scenario_regression=maximum_scenario_regression,
            minimum_mean_delta=minimum_mean_delta,
            maximum_latency_p95_ms=maximum_latency_p95_ms,
            maximum_memory_mb=maximum_memory_mb,
        )
        if reason is not None:
            excluded.append((evaluation.candidate_id, reason))
            continue
        analysis = analyses[evaluation.candidate_id]
        package = packages[evaluation.candidate_id]
        safe.append((evaluation, analysis, package))
    distinct: dict[
        tuple[float, ...],
        tuple[CandidateEvaluation, PairedAnalysis, CandidatePackage],
    ] = {}
    for item in safe:
        signature = tuple(round(value, 12) for value in item[0].session_rewards)
        previous = distinct.get(signature)
        if previous is None or _score_key(item) < _score_key(previous):
            if previous is not None:
                excluded.append(
                    (
                        previous[0].candidate_id,
                        f"behaviorally duplicates {item[0].candidate_id}",
                    )
                )
            distinct[signature] = item
        else:
            excluded.append(
                (
                    item[0].candidate_id,
                    f"behaviorally duplicates {previous[0].candidate_id}",
                )
            )
    safe = list(distinct.values())
    if len(safe) < 3:
        raise ValueError(
            "fewer than three distinct confirmed safe candidates remain: "
            + "; ".join(f"{name}={reason}" for name, reason in excluded)
        )

    score = min(safe, key=_score_key)
    remaining = [item for item in safe if item[0].candidate_id != score[0].candidate_id]
    robust = min(remaining, key=_robust_key)
    remaining = [
        item for item in remaining if item[0].candidate_id != robust[0].candidate_id
    ]
    efficient_pool = [
        item
        for item in remaining
        if item[0].score >= score[0].score - efficient_score_band
    ] or remaining
    efficient = min(efficient_pool, key=_efficiency_key)

    return TopThreeSelection(
        baseline_id=baseline_id,
        score_leader=_proposal(
            "score_leader",
            score,
            "highest validated technical score after readiness and safety gates",
        ),
        robust_leader=_proposal(
            "robust_leader",
            robust,
            "best paired lower bound, worst-scenario delta, and win/loss evidence among remaining candidates",
        ),
        efficient_alternative=_proposal(
            "efficient_alternative",
            efficient,
            "lowest measured latency, memory, and complexity inside the efficiency score band",
        ),
        excluded=tuple(excluded),
    )


def _proposal(
    role: ProposalRole,
    value: tuple[CandidateEvaluation, PairedAnalysis, CandidatePackage],
    reason: str,
) -> ProposalCandidate:
    return ProposalCandidate(role, value[0], value[1], value[2], reason)


def _score_key(
    value: tuple[CandidateEvaluation, PairedAnalysis, CandidatePackage],
) -> tuple[float, float, float, int, float, str]:
    evaluation, analysis, _ = value
    return (
        -evaluation.score,
        -analysis.mean_delta,
        -analysis.confidence_interval[0],
        evaluation.complexity,
        evaluation.latency_p95_ms,
        evaluation.candidate_id,
    )


def _robust_key(
    value: tuple[CandidateEvaluation, PairedAnalysis, CandidatePackage],
) -> tuple[float, float, int, float, float, int, str]:
    evaluation, analysis, _ = value
    worst_scenario = min(analysis.scenario_deltas.values(), default=0.0)
    return (
        -analysis.confidence_interval[0],
        -worst_scenario,
        -(analysis.wins - analysis.losses),
        -analysis.mean_delta,
        -evaluation.score,
        evaluation.complexity,
        evaluation.candidate_id,
    )


def _efficiency_key(
    value: tuple[CandidateEvaluation, PairedAnalysis, CandidatePackage],
) -> tuple[float, float, int, float, str]:
    evaluation, _, _ = value
    return (
        evaluation.latency_p95_ms,
        evaluation.memory_mb,
        evaluation.complexity,
        -evaluation.score,
        evaluation.candidate_id,
    )


def _exclusion_reason(
    evaluation: CandidateEvaluation,
    analysis: PairedAnalysis | None,
    package: CandidatePackage | None,
    *,
    baseline_id: str,
    root: Path,
    maximum_scenario_regression: float,
    minimum_mean_delta: float,
    maximum_latency_p95_ms: float,
    maximum_memory_mb: float,
) -> str | None:
    numeric = (
        evaluation.score,
        evaluation.latency_p95_ms,
        evaluation.memory_mb,
        *evaluation.session_rewards,
        *evaluation.scenario_scores.values(),
    )
    if not evaluation.session_rewards or not all(
        math.isfinite(item) for item in numeric
    ):
        return "missing or non-finite evaluation evidence"
    if not math.isclose(
        evaluation.score,
        statistics.fmean(evaluation.session_rewards),
        abs_tol=1e-6,
    ):
        return "score does not match mean paired session reward"
    if analysis is None:
        return "paired analysis missing"
    if analysis.candidate_id != evaluation.candidate_id:
        return "paired analysis candidate ID mismatch"
    if analysis.baseline_id != baseline_id:
        return "paired analysis baseline mismatch"
    if analysis.wins + analysis.ties + analysis.losses != len(
        evaluation.session_rewards
    ):
        return "paired counts do not match evaluated sessions"
    analysis_numeric = (
        analysis.mean_delta,
        *analysis.confidence_interval,
        analysis.randomization_pvalue,
        *analysis.scenario_deltas.values(),
    )
    if not all(math.isfinite(item) for item in analysis_numeric):
        return "non-finite paired analysis"
    if not 0.0 <= analysis.randomization_pvalue <= 1.0:
        return "invalid randomization p-value"
    if analysis.confidence_interval[0] > analysis.confidence_interval[1]:
        return "invalid confidence interval"
    if not (
        analysis.confidence_interval[0]
        <= analysis.mean_delta
        <= analysis.confidence_interval[1]
    ):
        return "paired mean lies outside confidence interval"
    if analysis.mean_delta < minimum_mean_delta:
        return "mean paired delta below declared minimum"
    if any(
        delta < -maximum_scenario_regression
        for delta in analysis.scenario_deltas.values()
    ):
        return "scenario safety regression"
    if evaluation.latency_p95_ms > maximum_latency_p95_ms:
        return "latency budget exceeded"
    if evaluation.memory_mb > maximum_memory_mb:
        return "memory budget exceeded"
    if package is None or not package.confirmed:
        return "candidate package is unconfirmed"
    if not package.safe:
        return "candidate package is marked unsafe"
    if package.candidate_id != evaluation.candidate_id:
        return "candidate package ID mismatch"
    for relative in (*package.assets, *package.evidence_refs):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return "package path escapes project root"
        if not path.exists():
            return f"required package path is missing: {relative}"
    return None


def _safe_project_path(value: str) -> None:
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError("package paths must stay inside the project")

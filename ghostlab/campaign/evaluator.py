from __future__ import annotations

import hashlib
import math
import resource
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from evaluator.local_evaluator import catalog_index, load_jsonl
from ghostlab.campaign.models import (
    CampaignJob,
    CandidateSpec,
    FidelityBudget,
    JobOutcome,
)
from ghostlab.competition.contract import AgentProtocol
from ghostlab.research.replay import evaluate_shared, session_reward
from ghostlab.training.protocol import FitReceipt

CandidateBuilder = Callable[[CandidateSpec], AgentProtocol]
FittedCandidateBuilder = Callable[
    [CandidateSpec, CampaignJob, tuple[str, ...]],
    tuple[AgentProtocol, tuple[FitReceipt, ...]],
]
_FORBIDDEN_PATH_MARKERS = ("f3", "holdout", "protected", "sealed")


def _require_development_path(path: Path) -> None:
    lowered = "/".join(path.parts).casefold()
    if any(marker in lowered for marker in _FORBIDDEN_PATH_MARKERS):
        raise ValueError(f"protected dataset path is forbidden in campaigns: {path}")


def _stratified_ids(
    samples: dict[str, dict], *, count: int, seed: int
) -> tuple[str, ...]:
    if count <= 0 or count >= len(samples):
        return tuple(sorted(samples))
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id, sample in samples.items():
        grouped[str(sample["scenario_type"])].append(sample_id)
    selected: list[str] = []
    remaining = count
    scenarios = sorted(grouped)
    for index, scenario in enumerate(scenarios):
        rows = sorted(
            grouped[scenario],
            key=lambda sample_id: hashlib.sha256(
                f"{seed}:{scenario}:{sample_id}".encode()
            ).hexdigest(),
        )
        allocation = (
            min(len(rows), max(0, remaining))
            if index + 1 == len(scenarios)
            else min(
                len(rows),
                max(0, remaining),
                round(count * len(rows) / max(1, len(samples))),
            )
        )
        selected.extend(rows[:allocation])
        remaining -= allocation
    if remaining > 0:
        unused = sorted(
            set(samples) - set(selected),
            key=lambda sample_id: hashlib.sha256(
                f"{seed}:remainder:{sample_id}".encode()
            ).hexdigest(),
        )
        selected.extend(unused[:remaining])
    return tuple(sorted(selected[:count]))


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


class _TimedAgent:
    def __init__(self, agent: AgentProtocol) -> None:
        self.agent = agent
        self.latencies_ms: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        result = self.agent.respond(session_id, user_message, turn, top_k)
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return result


class OfflineCampaignEvaluator:
    """Exact public-development replay adapter; protected paths are rejected."""

    def __init__(
        self,
        *,
        candidates: tuple[CandidateSpec, ...],
        builder: CandidateBuilder,
        fitted_builder: FittedCandidateBuilder | None = None,
        dataset_path: str | Path,
        catalog_path: str | Path,
        adaptive_sample_ids: tuple[str, ...],
        outer_folds: tuple[tuple[str, ...], ...],
        budgets: FidelityBudget,
        search_outer_folds: tuple[int, ...] | None = None,
        confirmation_outer_folds: tuple[int, ...] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.catalog_path = Path(catalog_path)
        _require_development_path(self.dataset_path)
        _require_development_path(self.catalog_path)
        self.builder = builder
        self.fitted_builder = fitted_builder
        self.budgets = budgets
        self.candidates = {item.canonical_hash(): item for item in candidates}
        if len(self.candidates) != len(candidates):
            raise ValueError("campaign candidates must have unique canonical hashes")
        adaptive = set(adaptive_sample_ids)
        rows = load_jsonl(self.dataset_path)
        self.samples = {
            str(row["sample_id"]): row
            for row in rows
            if str(row["sample_id"]) in adaptive
        }
        if set(self.samples) != adaptive:
            raise ValueError("adaptive split does not match the development dataset")
        self.outer_folds = outer_folds
        flattened = [item for fold in outer_folds for item in fold]
        if set(flattened) != adaptive or len(flattened) != len(set(flattened)):
            raise ValueError("outer folds must partition the adaptive split exactly")
        if (search_outer_folds is None) != (confirmation_outer_folds is None):
            raise ValueError("search and confirmation folds must be declared together")
        partitioned = search_outer_folds is not None
        self.search_outer_folds = (
            tuple(range(len(outer_folds)))
            if search_outer_folds is None
            else search_outer_folds
        )
        self.confirmation_outer_folds = (
            tuple(range(len(outer_folds)))
            if confirmation_outer_folds is None
            else confirmation_outer_folds
        )
        for label, folds in (
            ("search", self.search_outer_folds),
            ("confirmation", self.confirmation_outer_folds),
        ):
            if not folds or len(folds) != len(set(folds)):
                raise ValueError(f"{label} fold IDs must be non-empty and unique")
            if any(index < 0 or index >= len(outer_folds) for index in folds):
                raise ValueError(f"{label} fold ID is outside the nested split")
        if partitioned and set(self.search_outer_folds) & set(
            self.confirmation_outer_folds
        ):
            raise ValueError("search and confirmation folds must be disjoint")
        search_ids = {
            sample_id
            for index in self.search_outer_folds
            for sample_id in outer_folds[index]
        }
        self.search_samples = {
            sample_id: sample
            for sample_id, sample in self.samples.items()
            if sample_id in search_ids
        }
        _, self.categories, self.products = catalog_index(self.catalog_path)

    def _sample_ids(self, job: CampaignJob) -> tuple[str, ...]:
        if job.fidelity == "f2":
            if job.outer_fold not in self.confirmation_outer_folds:
                raise ValueError("F2 jobs require a frozen confirmation fold")
            assert job.outer_fold is not None
            return tuple(sorted(self.outer_folds[job.outer_fold]))
        count = self.budgets.f0 if job.fidelity == "f0" else self.budgets.f1
        return _stratified_ids(self.search_samples, count=count, seed=job.seed)

    def __call__(self, job: CampaignJob) -> JobOutcome:
        candidate = self.candidates.get(job.candidate_hash)
        if candidate is None:
            raise ValueError("campaign job references an unknown candidate")
        sample_ids = self._sample_ids(job)
        before_memory = _memory_mb()
        started = time.perf_counter()
        receipts: tuple[FitReceipt, ...] = ()
        if self.fitted_builder is None:
            agent = self.builder(candidate)
        else:
            agent, receipts = self.fitted_builder(candidate, job, sample_ids)
        timed = _TimedAgent(agent)
        result = evaluate_shared(
            timed,
            [self.samples[sample_id] for sample_id in sample_ids],
            self.categories,
            self.products,
            catalog_path=self.catalog_path,
            seed=job.seed,
        )
        elapsed = time.perf_counter() - started
        rewards = tuple(session_reward(item) for item in result["sessions"])
        scenarios: dict[str, list[float]] = defaultdict(list)
        for session, reward in zip(result["sessions"], rewards, strict=True):
            scenarios[str(session["scenario_type"])].append(reward)
        return JobOutcome(
            job_id=job.job_id,
            state="complete",
            score=float(result["recommended_technical_score"]),
            session_rewards=rewards,
            scenario_scores={
                name: statistics.fmean(values)
                for name, values in sorted(scenarios.items())
            },
            hit_rate_at_10=float(result["hit_rate_at_10"]),
            mrr=float(result["mrr"]),
            mttc=float(result["mttc"]),
            elapsed_seconds=elapsed,
            latency_p95_ms=_percentile_95(timed.latencies_ms),
            memory_mb=max(0.0, _memory_mb() - before_memory),
            fit_receipts=receipts,
        )

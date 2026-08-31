from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Fidelity = Literal["f0", "f1", "f2"]
Strategy = Literal["random", "grid", "beam", "allocated"]
Objective = Callable[["PolicyCandidate", Fidelity, int], float]


class PolicyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    family: str
    techniques: tuple[str, ...] = ()
    parameters: tuple[tuple[str, str | int | float | bool], ...] = ()
    complexity: int = Field(default=1, ge=0)

    def canonical_hash(self) -> str:
        payload = self.model_dump(mode="json")
        payload["techniques"] = sorted(payload["techniques"])
        payload["parameters"] = sorted(payload["parameters"])
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class Evaluation:
    candidate: str
    candidate_hash: str
    fidelity: Fidelity
    score: float
    ordinal: int


@dataclass(frozen=True)
class SearchResult:
    strategy: Strategy
    seed: int
    budget: int
    winner: str
    winner_score: float
    evaluations: tuple[Evaluation, ...]


def _unique(candidates: Iterable[PolicyCandidate]) -> list[PolicyCandidate]:
    result: list[PolicyCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.canonical_hash()
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def search(
    candidates: Iterable[PolicyCandidate],
    objective: Objective,
    *,
    strategy: Strategy,
    budget: int,
    seed: int,
) -> SearchResult:
    if budget <= 0:
        raise ValueError("budget must be positive")
    pool = _unique(candidates)
    if not pool:
        raise ValueError("candidate pool cannot be empty")
    rng = random.Random(seed)
    schedule: list[tuple[PolicyCandidate, Fidelity]]
    if strategy == "random":
        rng.shuffle(pool)
        schedule = [(candidate, "f2") for candidate in pool]
    elif strategy == "grid":
        schedule = [
            (candidate, "f2") for candidate in sorted(pool, key=lambda item: item.name)
        ]
    else:
        if strategy == "allocated":
            by_family: dict[str, list[PolicyCandidate]] = {}
            for candidate in pool:
                by_family.setdefault(candidate.family, []).append(candidate)
            ordered: list[PolicyCandidate] = []
            while any(by_family.values()):
                for family in sorted(by_family):
                    if by_family[family]:
                        ordered.append(by_family[family].pop(0))
        else:
            ordered = sorted(pool, key=lambda item: (item.complexity, item.name))
        screen_count = min(len(ordered), max(1, budget // 2))
        screen_scores = [
            (candidate, objective(candidate, "f0", seed))
            for candidate in ordered[:screen_count]
        ]
        screen_evaluations = [
            Evaluation(
                candidate.name,
                candidate.canonical_hash(),
                "f0",
                score,
                ordinal,
            )
            for ordinal, (candidate, score) in enumerate(screen_scores, start=1)
        ]
        ranked = sorted(
            screen_scores, key=lambda item: (-item[1], item[0].complexity, item[0].name)
        )
        for candidate, _ in ranked:
            if len(screen_evaluations) >= budget:
                break
            screen_evaluations.append(
                Evaluation(
                    candidate.name,
                    candidate.canonical_hash(),
                    "f2",
                    objective(candidate, "f2", seed),
                    len(screen_evaluations) + 1,
                )
            )
        eligible = [
            item for item in screen_evaluations if item.fidelity == "f2"
        ] or screen_evaluations
        winner_eval = min(eligible, key=lambda item: (-item.score, item.candidate))
        return SearchResult(
            strategy=strategy,
            seed=seed,
            budget=budget,
            winner=winner_eval.candidate,
            winner_score=winner_eval.score,
            evaluations=tuple(screen_evaluations),
        )

    evaluations: list[Evaluation] = []
    for candidate, fidelity in schedule:
        if len(evaluations) >= budget:
            break
        key = candidate.canonical_hash()
        score = objective(candidate, fidelity, seed)
        evaluations.append(
            Evaluation(candidate.name, key, fidelity, score, len(evaluations) + 1)
        )
    full = [item for item in evaluations if item.fidelity == "f2"]
    eligible = full or evaluations
    winner_eval = min(eligible, key=lambda item: (-item.score, item.candidate))
    return SearchResult(
        strategy=strategy,
        seed=seed,
        budget=budget,
        winner=winner_eval.candidate,
        winner_score=winner_eval.score,
        evaluations=tuple(evaluations),
    )


def save_checkpoint(path: Path, result: SearchResult) -> None:
    payload = {
        **asdict(result),
        "evaluations": [asdict(item) for item in result.evaluations],
    }
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def interaction_gain(
    base: float, first: float, second: float, combination: float
) -> float:
    return combination - first - second + base


def should_retest(
    standalone_delta: float, interaction: float, threshold: float = 0.01
) -> bool:
    return standalone_delta < 0 and interaction >= threshold

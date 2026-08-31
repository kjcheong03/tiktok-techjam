from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Trial:
    trial_id: str
    parameters: tuple[tuple[str, str | int | float | bool], ...]


@dataclass(frozen=True)
class TrialResult:
    trial: Trial
    resource: int
    score: float


Objective = Callable[[Trial, int], float]


def successive_halving(
    trials: tuple[Trial, ...],
    objective: Objective,
    *,
    resources: tuple[int, ...],
    reduction_factor: int = 3,
) -> tuple[TrialResult, ...]:
    """Deterministic successive halving for an already frozen trial population."""

    if not trials or not resources:
        raise ValueError("trials and resource levels cannot be empty")
    if reduction_factor < 2:
        raise ValueError("reduction_factor must be at least two")
    if any(value <= 0 for value in resources) or tuple(sorted(resources)) != resources:
        raise ValueError("resource levels must be positive and increasing")
    survivors = sorted(trials, key=lambda item: item.trial_id)
    outcomes: list[TrialResult] = []
    for index, resource in enumerate(resources):
        level = [
            TrialResult(trial, resource, float(objective(trial, resource)))
            for trial in survivors
        ]
        outcomes.extend(level)
        if index + 1 < len(resources):
            keep = max(1, len(level) // reduction_factor)
            survivors = [
                item.trial
                for item in sorted(
                    level, key=lambda item: (-item.score, item.trial.trial_id)
                )[:keep]
            ]
    return tuple(outcomes)

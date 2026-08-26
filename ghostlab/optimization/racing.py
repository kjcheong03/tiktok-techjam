from __future__ import annotations

import statistics
from typing import Literal

from ghostlab.evaluation.statistics import bootstrap_mean_interval

Decision = Literal["PROMOTE", "REJECT", "HOLD_MORE_DATA", "NOVELTY_RESERVE"]


def racing_decide(
    deltas: list[float],
    *,
    fidelity: Literal["f0", "f1", "f2"],
    behavior_novelty: float = 0.0,
    catastrophic_threshold: float = -0.15,
    material_delta: float = 0.01,
    seed: int = 20260826,
) -> Decision:
    if not deltas:
        raise ValueError("paired deltas cannot be empty")
    mean = statistics.fmean(deltas)
    if mean <= catastrophic_threshold:
        return "REJECT"
    if fidelity == "f0":
        if behavior_novelty >= 0.5 and mean > -material_delta:
            return "NOVELTY_RESERVE"
        return "PROMOTE" if mean >= -material_delta else "HOLD_MORE_DATA"
    resamples = 1000 if fidelity == "f1" else 5000
    confidence = 0.80 if fidelity == "f1" else 0.95
    lower, upper = bootstrap_mean_interval(
        deltas, resamples=resamples, confidence=confidence, seed=seed
    )
    if lower > 0.0 and mean >= material_delta:
        return "PROMOTE"
    if upper < -material_delta:
        return "REJECT"
    if behavior_novelty >= 0.5 and mean > -material_delta:
        return "NOVELTY_RESERVE"
    return "HOLD_MORE_DATA"

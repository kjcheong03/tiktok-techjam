from __future__ import annotations

import random
import statistics
from collections.abc import Sequence


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    resamples: int,
    confidence: float = 0.95,
    seed: int = 20260826,
) -> tuple[float, float]:
    if not values or resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap inputs")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = estimates[max(0, int(tail * resamples))]
    upper = estimates[min(resamples - 1, int((1.0 - tail) * resamples))]
    return lower, upper


def paired_randomization_pvalue(
    values: Sequence[float], *, resamples: int, seed: int = 20260826
) -> float:
    if not values or resamples <= 0:
        raise ValueError("invalid randomization inputs")
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(resamples):
        permuted = abs(
            statistics.fmean(
                value if rng.random() < 0.5 else -value for value in values
            )
        )
        exceedances += permuted >= observed
    return (exceedances + 1) / (resamples + 1)

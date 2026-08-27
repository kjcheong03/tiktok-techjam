from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: Literal["float", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[str | int | float | bool, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError("categorical parameters require choices")
        elif self.low is None or self.high is None or self.low > self.high:
            raise ValueError("numeric parameters require ordered bounds")


@dataclass(frozen=True)
class Observation:
    parameters: tuple[tuple[str, str | int | float | bool], ...]
    score: float


def _random_value(parameter: Parameter, rng: random.Random) -> str | int | float | bool:
    if parameter.kind == "categorical":
        return rng.choice(parameter.choices)
    assert parameter.low is not None and parameter.high is not None
    if parameter.kind == "int":
        return rng.randint(int(parameter.low), int(parameter.high))
    return rng.uniform(float(parameter.low), float(parameter.high))


def suggest(
    space: tuple[Parameter, ...],
    observations: tuple[Observation, ...],
    *,
    seed: int,
    exploration_fraction: float = 0.2,
    top_fraction: float = 0.25,
) -> tuple[tuple[str, str | int | float | bool], ...]:
    """Small BOHB-style sampler: random exploration plus elite-local proposals."""

    if not space:
        raise ValueError("parameter space cannot be empty")
    if not 0.0 <= exploration_fraction <= 1.0 or not 0.0 < top_fraction <= 1.0:
        raise ValueError("invalid BOHB fractions")
    rng = random.Random(seed)
    explore = not observations or rng.random() < exploration_fraction
    elite: list[Observation] = []
    if not explore:
        count = max(1, round(len(observations) * top_fraction))
        elite = sorted(observations, key=lambda item: -item.score)[:count]
    values: list[tuple[str, str | int | float | bool]] = []
    for parameter in sorted(space, key=lambda item: item.name):
        if explore:
            value = _random_value(parameter, rng)
        else:
            source = rng.choice(elite)
            source_values = dict(source.parameters)
            center = source_values.get(parameter.name)
            if parameter.kind == "categorical" or not isinstance(center, (int, float)):
                value = center if center is not None else _random_value(parameter, rng)
            else:
                assert parameter.low is not None and parameter.high is not None
                width = (float(parameter.high) - float(parameter.low)) * 0.15
                sampled = min(
                    float(parameter.high),
                    max(
                        float(parameter.low),
                        rng.gauss(float(center), max(width, 1e-12)),
                    ),
                )
                value = round(sampled) if parameter.kind == "int" else sampled
        values.append((parameter.name, value))
    return tuple(values)

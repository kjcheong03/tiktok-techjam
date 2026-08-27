from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ghostlab.optimization.bohb import Observation, Parameter, suggest


class ConditionalParameter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: Literal["float", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[str | int | float | bool, ...] = ()
    requires_all: tuple[str, ...] = ()
    requires_any: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_domain(self) -> ConditionalParameter:
        Parameter(self.name, self.kind, self.low, self.high, self.choices)
        return self

    def eligible(self, techniques: frozenset[str]) -> bool:
        return set(self.requires_all) <= techniques and (
            not self.requires_any or bool(set(self.requires_any) & techniques)
        )

    def parameter(self) -> Parameter:
        return Parameter(self.name, self.kind, self.low, self.high, self.choices)


class ConditionalSearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    parameters: tuple[ConditionalParameter, ...]

    @model_validator(mode="after")
    def unique_names(self) -> ConditionalSearchSpace:
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("conditional parameter names must be unique")
        return self

    def for_techniques(self, technique_ids: tuple[str, ...]) -> tuple[Parameter, ...]:
        selected = frozenset(technique_ids)
        return tuple(
            item.parameter() for item in self.parameters if item.eligible(selected)
        )


class TuningContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outer_fold: int = Field(ge=0)
    inner_fold: int = Field(ge=0)
    split_role: Literal["inner_validation"] = "inner_validation"


def suggest_for_combination(
    space: ConditionalSearchSpace,
    technique_ids: tuple[str, ...],
    observations: tuple[Observation, ...],
    *,
    context: TuningContext,
    seed: int,
    exploration_fraction: float = 0.2,
) -> tuple[tuple[str, str | int | float | bool], ...]:
    del context  # its validated type is the leakage boundary
    eligible = space.for_techniques(technique_ids)
    if not eligible:
        return ()
    return suggest(
        eligible,
        observations,
        seed=seed,
        exploration_fraction=exploration_fraction,
    )


def two_way_simplex_grid(step: float) -> tuple[tuple[float, float], ...]:
    if not 0.0 < step <= 1.0:
        raise ValueError("simplex step must be in (0, 1]")
    count = round(1.0 / step)
    if abs(count * step - 1.0) > 1e-9:
        raise ValueError("simplex step must divide one exactly")
    return tuple((index * step, 1.0 - index * step) for index in range(count + 1))

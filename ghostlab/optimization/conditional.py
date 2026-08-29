from __future__ import annotations

import math
import random
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
    scale: Literal["linear", "log"] = "linear"
    requires_all: tuple[str, ...] = ()
    requires_any: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_domain(self) -> ConditionalParameter:
        Parameter(self.name, self.kind, self.low, self.high, self.choices, self.scale)
        return self

    def eligible(self, techniques: frozenset[str]) -> bool:
        return set(self.requires_all) <= techniques and (
            not self.requires_any or bool(set(self.requires_any) & techniques)
        )

    def parameter(self) -> Parameter:
        return Parameter(
            self.name, self.kind, self.low, self.high, self.choices, self.scale
        )


class ConditionalSearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2] = 1
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
    center: tuple[tuple[str, str | int | float | bool], ...] | None = None,
    max_changes: int = 3,
    trust_region: float = 0.2,
    block_index: int = 0,
) -> tuple[tuple[str, str | int | float | bool], ...]:
    del context  # its validated type is the leakage boundary
    eligible = space.for_techniques(technique_ids)
    if not eligible:
        return ()
    if center is not None:
        return _suggest_local(
            eligible,
            center,
            observations,
            seed=seed,
            max_changes=max_changes,
            trust_region=trust_region,
            block_index=block_index,
        )
    return suggest(
        eligible,
        observations,
        seed=seed,
        exploration_fraction=exploration_fraction,
    )


_SPARSE_FIELDS = (
    "sparse_title_weight",
    "sparse_categories_weight",
    "sparse_features_weight",
    "sparse_details_weight",
    "sparse_store_weight",
    "sparse_description_weight",
)


def _parameter_block(name: str) -> str:
    if name.startswith(("question_", "eig_")):
        return "dialogue"
    if name.startswith(
        (
            "retrieval_",
            "sparse_",
            "rrf_",
            "fusion_",
            "dense_",
            "expansion_",
            "query_expansion_",
            "semantic_rescue_",
        )
    ):
        return "retrieval"
    return "ranking"


def _local_value(
    parameter: Parameter,
    center: str | float | bool,
    *,
    rng: random.Random,
    trust_region: float,
) -> str | int | float | bool:
    if parameter.kind == "categorical":
        alternatives = tuple(item for item in parameter.choices if item != center)
        return rng.choice(alternatives or parameter.choices)
    assert parameter.low is not None and parameter.high is not None
    numeric_center = float(center)
    if parameter.scale == "log":
        low = math.log(float(parameter.low))
        high = math.log(float(parameter.high))
        center_log = min(high, max(low, math.log(max(numeric_center, 1e-12))))
        radius = max((high - low) * trust_region, 1e-12)
        sampled = math.exp(rng.uniform(max(low, center_log - radius), min(high, center_log + radius)))
    else:
        low = float(parameter.low)
        high = float(parameter.high)
        radius = max((high - low) * trust_region, 1e-12)
        sampled = rng.uniform(max(low, numeric_center - radius), min(high, numeric_center + radius))
    return round(sampled) if parameter.kind == "int" else sampled


def _suggest_local(
    space: tuple[Parameter, ...],
    center: tuple[tuple[str, str | int | float | bool], ...],
    observations: tuple[Observation, ...],
    *,
    seed: int,
    max_changes: int,
    trust_region: float,
    block_index: int,
) -> tuple[tuple[str, str | int | float | bool], ...]:
    """Mutate one coherent parameter block near an observed effective config."""

    if not 1 <= max_changes <= 3:
        raise ValueError("local HPO must change between one and three groups")
    if not 0.0 < trust_region <= 0.5:
        raise ValueError("local HPO trust region must be in (0, 0.5]")
    values = dict(center)
    if observations:
        elite = max(observations, key=lambda item: item.score)
        values.update(elite.parameters)
    by_name = {item.name: item for item in space if item.name in values}
    space = tuple(item for item in space if item.name in values)
    if not space:
        return ()
    blocks = tuple(
        block
        for block in ("ranking", "retrieval", "dialogue")
        if any(_parameter_block(item.name) == block for item in space)
    )
    if not blocks:
        return ()
    selected_block = blocks[block_index % len(blocks)]
    names = sorted(
        item.name for item in space if _parameter_block(item.name) == selected_block
    )
    sparse = tuple(name for name in _SPARSE_FIELDS if name in names)
    units: list[tuple[str, ...]] = [(name,) for name in names if name not in sparse]
    if sparse:
        units.append(sparse)
    rng = random.Random(seed)
    rng.shuffle(units)
    chosen = units[: min(max_changes, len(units))]
    categorical = next(
        (
            unit
            for unit in chosen
            if len(unit) == 1 and by_name[unit[0]].kind == "categorical"
        ),
        None,
    )
    if categorical is not None:
        chosen = [categorical]
    result: dict[str, str | int | float | bool] = {}
    for unit in chosen:
        if unit == sparse and sparse:
            original = [float(values[name]) for name in sparse]
            sampled = [
                float(
                    _local_value(
                        by_name[name],
                        values[name],
                        rng=rng,
                        trust_region=trust_region,
                    )
                )
                for name in sparse
            ]
            target_sum = sum(original)
            sampled_sum = sum(sampled)
            normalized = (
                [value * target_sum / sampled_sum for value in sampled]
                if target_sum > 0.0 and sampled_sum > 0.0
                else sampled
            )
            result.update(zip(sparse, normalized, strict=True))
            continue
        name = unit[0]
        result[name] = _local_value(
            by_name[name], values[name], rng=rng, trust_region=trust_region
        )
    return tuple(sorted(result.items()))


def two_way_simplex_grid(step: float) -> tuple[tuple[float, float], ...]:
    if not 0.0 < step <= 1.0:
        raise ValueError("simplex step must be in (0, 1]")
    count = round(1.0 / step)
    if abs(count * step - 1.0) > 1e-9:
        raise ValueError("simplex step must divide one exactly")
    return tuple((index * step, 1.0 - index * step) for index in range(count + 1))

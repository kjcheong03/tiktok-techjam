from __future__ import annotations

import itertools
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ghostlab.retrieval.gbdt import GBDTFeatureStore, LambdaMARTModel
from ghostlab.retrieval.reward_lambdamart import mean_predicted_terminal_reward

RankAggregation = Literal["mean_rank", "median_rank", "reciprocal_rank"]
ModelAggregation = Literal["standardized_score", "mean_rank"]


def aggregate_rankings(
    rankings: list[list[str]] | tuple[list[str], ...],
    *,
    method: RankAggregation = "mean_rank",
    weights: tuple[float, ...] | None = None,
    reciprocal_k: float = 60.0,
) -> list[str]:
    """Combine already-depth-matched rankings with stable deterministic ties."""
    if not rankings:
        return []
    if any(len(set(ranking)) != len(ranking) for ranking in rankings):
        raise ValueError("input rankings must not contain duplicate identifiers")
    normalized = _normalized_weights(len(rankings), weights)
    identifiers = list(dict.fromkeys(itertools.chain.from_iterable(rankings)))
    positions = [
        {identifier: rank for rank, identifier in enumerate(ranking, 1)}
        for ranking in rankings
    ]
    missing_ranks = [len(ranking) + 1 for ranking in rankings]

    def score(identifier: str) -> float:
        ranks = [
            table.get(identifier, missing)
            for table, missing in zip(positions, missing_ranks, strict=True)
        ]
        if method == "mean_rank":
            return -sum(weight * rank for weight, rank in zip(normalized, ranks))
        if method == "median_rank":
            return -statistics.median(ranks)
        if reciprocal_k < 0.0:
            raise ValueError("reciprocal_k cannot be negative")
        return sum(
            weight / (reciprocal_k + rank)
            for weight, rank in zip(normalized, ranks, strict=True)
        )

    first_seen = {identifier: index for index, identifier in enumerate(identifiers)}
    return sorted(identifiers, key=lambda item: (-score(item), first_seen[item], item))


class ModelRankEnsembleReranker:
    """Compact offline ensemble over compatible fold/objective tree assets."""

    def __init__(
        self,
        features: GBDTFeatureStore,
        models: tuple[LambdaMARTModel, ...],
        *,
        method: ModelAggregation = "standardized_score",
        weights: tuple[float, ...] | None = None,
    ) -> None:
        if not models:
            raise ValueError("an ensemble needs at least one model")
        feature_names = models[0].feature_names
        if any(model.feature_names != feature_names for model in models):
            raise ValueError("ensemble models must use identical runtime features")
        self.features = features
        self.models = models
        self.method = method
        self.weights = _normalized_weights(len(models), weights)

    @classmethod
    def from_asset(
        cls,
        features: GBDTFeatureStore,
        asset: RankEnsembleAsset,
        *,
        project_root: str | Path,
    ) -> ModelRankEnsembleReranker:
        root = Path(project_root).resolve()
        models = tuple(
            LambdaMARTModel.load(root / relative) for relative in asset.model_assets
        )
        return cls(features, models, method=asset.aggregation, weights=asset.weights)

    def rerank(
        self, query: str, ranking: list[str], *, rerank_k: int = 50
    ) -> list[str]:
        head = ranking[:rerank_k]
        if len(head) < 2:
            return list(ranking)
        matrix = self.features.matrix(query, head, self.models[0].feature_names)
        head_scores = np.vstack([model.predict(matrix) for model in self.models])
        combined = combine_model_scores(
            head_scores, method=self.method, weights=self.weights
        )
        ordered = np.argsort(-combined, kind="stable")
        return [*[head[index] for index in ordered], *ranking[rerank_k:]]


@dataclass(frozen=True)
class RankStackAsset:
    technique_id: str
    weights: tuple[float, ...]
    grid_step: float
    inner_validation_reward: float


@dataclass(frozen=True)
class RankEnsembleAsset:
    technique_id: str
    aggregation: ModelAggregation
    model_assets: tuple[str, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.technique_id not in {
            "ranking.fold_ensemble.v1",
            "fusion.rank_stack.v1",
        }:
            raise ValueError("unknown ensemble technique ID")
        if not self.model_assets:
            raise ValueError("ensemble asset must name at least one model")
        for value in self.model_assets:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("model assets must be safe project-relative paths")
        _normalized_weights(len(self.model_assets), self.weights)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> RankEnsembleAsset:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        value["model_assets"] = tuple(value["model_assets"])
        value["weights"] = tuple(float(item) for item in value["weights"])
        return cls(**value)


def fit_rank_stack_weights(
    head_scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: list[int],
    turns: list[int],
    *,
    grid_step: float = 0.25,
) -> RankStackAsset:
    """Select non-negative stack weights using fold-local validation only."""
    if head_scores.ndim != 2 or head_scores.shape[1] != len(labels):
        raise ValueError("head score matrix must be heads by candidate rows")
    if not 0.0 < grid_step <= 1.0 or not math.isclose(
        round(1.0 / grid_step) * grid_step, 1.0, abs_tol=1e-9
    ):
        raise ValueError("grid_step must divide one")
    units = round(1.0 / grid_step)
    candidates = _simplex_weights(head_scores.shape[0], units)
    standardized = _standardize_by_group(head_scores, groups)
    best_weights = candidates[0]
    best_reward = -math.inf
    for weight_units in candidates:
        weights = tuple(value / units for value in weight_units)
        combined = np.average(standardized, axis=0, weights=weights)
        reward = mean_predicted_terminal_reward(labels, combined, groups, turns)
        if reward > best_reward + 1e-12:
            best_reward = reward
            best_weights = weight_units
    return RankStackAsset(
        technique_id="fusion.rank_stack.v1",
        weights=tuple(value / units for value in best_weights),
        grid_step=grid_step,
        inner_validation_reward=best_reward,
    )


def combine_model_scores(
    head_scores: NDArray[np.float64],
    *,
    method: ModelAggregation,
    weights: tuple[float, ...] | None = None,
) -> NDArray[np.float64]:
    if head_scores.ndim != 2 or head_scores.shape[0] == 0:
        raise ValueError("head_scores must be a non-empty 2D matrix")
    normalized = _normalized_weights(head_scores.shape[0], weights)
    if method == "standardized_score":
        rows = []
        for row in head_scores:
            deviation = float(np.std(row))
            rows.append((row - np.mean(row)) / (deviation if deviation > 1e-12 else 1.0))
        return np.average(np.vstack(rows), axis=0, weights=normalized)
    ranks = np.empty_like(head_scores)
    for index, row in enumerate(head_scores):
        order = np.argsort(-row, kind="stable")
        ranks[index, order] = np.arange(1, len(row) + 1)
    return -np.average(ranks, axis=0, weights=normalized)


def _standardize_by_group(
    head_scores: NDArray[np.float64], groups: list[int]
) -> NDArray[np.float64]:
    if sum(groups) != head_scores.shape[1]:
        raise ValueError("group sizes do not align with score columns")
    result = np.empty_like(head_scores, dtype=np.float64)
    start = 0
    for size in groups:
        segment = head_scores[:, start : start + size]
        for head, row in enumerate(segment):
            deviation = float(np.std(row))
            result[head, start : start + size] = (row - np.mean(row)) / (
                deviation if deviation > 1e-12 else 1.0
            )
        start += size
    return result


def _simplex_weights(heads: int, units: int) -> list[tuple[int, ...]]:
    if heads <= 0:
        raise ValueError("heads must be positive")
    if heads == 1:
        return [(units,)]
    return [
        (first, *rest)
        for first in range(units + 1)
        for rest in _simplex_weights(heads - 1, units - first)
    ]


def _normalized_weights(
    count: int, weights: tuple[float, ...] | None
) -> tuple[float, ...]:
    if weights is None:
        return (1.0 / count,) * count
    if len(weights) != count or any(weight < 0.0 for weight in weights):
        raise ValueError("weights must be non-negative and align with inputs")
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    return tuple(weight / total for weight in weights)

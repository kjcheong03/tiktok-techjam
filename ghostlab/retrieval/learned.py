from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import query_terms

FEATURE_NAMES = (
    "title_overlap",
    "category_overlap",
    "feature_overlap",
    "details_overlap",
    "store_overlap",
    "description_overlap",
    "catalog_quality",
)
FIELD_FEATURE_NAMES = FEATURE_NAMES[:6]
CATALOG_FIELDS = ("title", "categories", "features", "details", "store", "description")


@dataclass(frozen=True)
class PairwiseExample:
    base_margin: float
    feature_delta: tuple[float, ...]


@dataclass(frozen=True)
class LinearRerankerModel:
    weights: tuple[float, ...]
    l2: float
    training_pairs: int

    def __post_init__(self) -> None:
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError("learned reranker feature count mismatch")


class CandidateFeatureStore:
    def __init__(
        self,
        catalog_path: str | Path,
        *,
        enabled_features: tuple[str, ...] = FEATURE_NAMES,
        quality: dict[str, float] | None = None,
    ) -> None:
        unknown = set(enabled_features) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown candidate features: {sorted(unknown)}")
        enabled = set(enabled_features)
        self.fields: dict[str, tuple[frozenset[str], ...]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                self.fields[str(product["parent_asin"])] = tuple(
                    (
                        frozenset(query_terms(str(product.get(field) or ""), 300))
                        if feature_name in enabled
                        else frozenset()
                    )
                    for field, feature_name in zip(
                        CATALOG_FIELDS, FIELD_FEATURE_NAMES, strict=True
                    )
                )
        if "catalog_quality" not in enabled:
            self.quality = {}
        elif quality is not None:
            self.quality = quality
        else:
            self.quality = CatalogQualityReranker(catalog_path).quality

    def features(self, query: str, identifier: str) -> tuple[float, ...]:
        query_set = frozenset(query_terms(query, 80))
        denominator = max(1, len(query_set))
        fields = self.fields.get(identifier, (frozenset(),) * 6)
        overlaps = tuple(len(query_set & terms) / denominator for terms in fields)
        return (*overlaps, self.quality.get(identifier, 0.0))


class LearnedLinearReranker:
    def __init__(
        self,
        features: CandidateFeatureStore,
        model: LinearRerankerModel,
    ) -> None:
        self.features = features
        self.model = model

    def rerank(
        self, query: str, ranking: list[str], *, rerank_k: int = 50
    ) -> list[str]:
        head = ranking[:rerank_k]
        count = len(head)
        weights: NDArray[np.float64] = np.asarray(self.model.weights, dtype=np.float64)
        original_ranks = {identifier: rank for rank, identifier in enumerate(head, 1)}
        scores = {}
        for rank, identifier in enumerate(head, 1):
            base = 1.0 if count == 1 else 1.0 - (rank - 1) / max(1, count - 1)
            values: NDArray[np.float64] = np.asarray(
                self.features.features(query, identifier), dtype=np.float64
            )
            scores[identifier] = base + float(values @ weights)
        ordered = sorted(
            head,
            key=lambda identifier: (
                -scores[identifier],
                original_ranks[identifier],
                identifier,
            ),
        )
        return [*ordered, *ranking[rerank_k:]]


def fit_pairwise_linear(
    examples: list[PairwiseExample],
    *,
    l2: float = 0.1,
    learning_rate: float = 0.5,
    iterations: int = 500,
    enabled_features: tuple[str, ...] = FEATURE_NAMES,
) -> LinearRerankerModel:
    if not examples:
        raise ValueError("pairwise training examples cannot be empty")
    if l2 < 0.0 or learning_rate <= 0.0 or iterations <= 0:
        raise ValueError("invalid optimizer settings")
    matrix: NDArray[np.float64] = np.asarray(
        [item.feature_delta for item in examples], dtype=np.float64
    )
    base: NDArray[np.float64] = np.asarray(
        [item.base_margin for item in examples], dtype=np.float64
    )
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("pairwise example feature count mismatch")
    unknown = set(enabled_features) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown learned features: {sorted(unknown)}")
    enabled: NDArray[np.float64] = np.asarray(
        [name in enabled_features for name in FEATURE_NAMES], dtype=np.float64
    )
    matrix *= enabled
    weights: NDArray[np.float64] = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for _ in range(iterations):
        margins = np.clip(base + matrix @ weights, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(margins))
        gradient = -(matrix.T @ probabilities) / len(examples) + l2 * weights
        weights -= learning_rate * gradient
    return LinearRerankerModel(
        weights=tuple(float(value) for value in weights),
        l2=l2,
        training_pairs=len(examples),
    )

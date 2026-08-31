from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ghostlab.retrieval.learned import CATALOG_FIELDS, FIELD_FEATURE_NAMES
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.sparse import query_terms

RANK_FEATURES = ("original_rank", "rank_percentile", "reciprocal_rank")
LEXICAL_FEATURES = (
    *RANK_FEATURES,
    "query_token_count",
    *FIELD_FEATURE_NAMES,
    *(f"{name}_missing" for name in FIELD_FEATURE_NAMES),
    "catalog_quality",
)
METADATA_FEATURES = (
    *LEXICAL_FEATURES,
    "average_rating",
    "average_rating_missing",
    "log_rating_number",
    "rating_number_missing",
    "metadata_completeness",
)
FEATURE_SETS = {
    "rank_only": RANK_FEATURES,
    "lexical": LEXICAL_FEATURES,
    "metadata": METADATA_FEATURES,
}


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


@dataclass(frozen=True)
class CatalogGBDTFeatures:
    field_terms: tuple[frozenset[str], ...]
    field_missing: tuple[bool, ...]
    average_rating: float | None
    rating_number: float | None
    metadata_completeness: float


class GBDTFeatureStore:
    """Deterministic runtime-safe candidate features with explicit missingness."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        quality: dict[str, float] | None = None,
    ) -> None:
        self.products: dict[str, CatalogGBDTFeatures] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                present = tuple(
                    _has_value(product.get(field)) for field in CATALOG_FIELDS
                )
                rating = product.get("average_rating")
                rating_number = product.get("rating_number")
                metadata_values = (
                    *present,
                    _has_value(product.get("average_rating")),
                    _has_value(product.get("rating_number")),
                    _has_value(product.get("price")),
                )
                self.products[str(product["parent_asin"])] = CatalogGBDTFeatures(
                    field_terms=tuple(
                        frozenset(query_terms(str(product.get(field) or ""), 300))
                        for field in CATALOG_FIELDS
                    ),
                    field_missing=tuple(not value for value in present),
                    average_rating=(
                        float(rating) if isinstance(rating, (int, float)) else None
                    ),
                    rating_number=(
                        float(rating_number)
                        if isinstance(rating_number, (int, float))
                        else None
                    ),
                    metadata_completeness=sum(metadata_values) / len(metadata_values),
                )
        self.quality = (
            quality
            if quality is not None
            else CatalogQualityReranker(catalog_path).quality
        )

    def all_features(
        self,
        query: str,
        identifier: str,
        *,
        rank: int,
        count: int,
    ) -> dict[str, float]:
        if rank <= 0 or count <= 0 or rank > count:
            raise ValueError("candidate rank must be within the ranking")
        query_set = frozenset(query_terms(query, 80))
        denominator = max(1, len(query_set))
        product = self.products.get(identifier)
        if product is None:
            field_terms: tuple[frozenset[str], ...] = (frozenset(),) * len(
                FIELD_FEATURE_NAMES
            )
            field_missing = (True,) * len(FIELD_FEATURE_NAMES)
            average_rating = rating_number = None
            completeness = 0.0
        else:
            field_terms = product.field_terms
            field_missing = product.field_missing
            average_rating = product.average_rating
            rating_number = product.rating_number
            completeness = product.metadata_completeness
        percentile = 0.0 if count == 1 else (rank - 1) / (count - 1)
        values: dict[str, float] = {
            "original_rank": float(rank),
            "rank_percentile": percentile,
            "reciprocal_rank": 1.0 / rank,
            "query_token_count": float(len(query_set)),
            "catalog_quality": self.quality.get(identifier, math.nan),
            "average_rating": (
                average_rating if average_rating is not None else math.nan
            ),
            "average_rating_missing": float(average_rating is None),
            "log_rating_number": (
                math.log1p(max(0.0, rating_number))
                if rating_number is not None
                else math.nan
            ),
            "rating_number_missing": float(rating_number is None),
            "metadata_completeness": completeness,
        }
        for name, terms, missing in zip(
            FIELD_FEATURE_NAMES, field_terms, field_missing, strict=True
        ):
            values[name] = len(query_set & terms) / denominator
            values[f"{name}_missing"] = float(missing)
        return values

    def matrix(
        self,
        query: str,
        ranking: list[str] | tuple[str, ...],
        feature_names: tuple[str, ...],
    ) -> NDArray[np.float64]:
        unknown = set(feature_names) - set(METADATA_FEATURES)
        if unknown:
            raise ValueError(f"unknown GBDT features: {sorted(unknown)}")
        count = len(ranking)
        if count == 0:
            return np.empty((0, len(feature_names)), dtype=np.float64)
        return np.asarray(
            [
                [values[name] for name in feature_names]
                for rank, identifier in enumerate(ranking, 1)
                for values in (
                    self.all_features(query, identifier, rank=rank, count=count),
                )
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RegressionTree:
    children_left: tuple[int, ...]
    children_right: tuple[int, ...]
    feature: tuple[int, ...]
    threshold: tuple[float, ...]
    missing_go_to_left: tuple[bool, ...]
    value: tuple[float, ...]

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        predictions: NDArray[np.float64] = np.empty(len(features), dtype=np.float64)
        for row_index, row in enumerate(features):
            node = 0
            while self.children_left[node] != self.children_right[node]:
                feature_index = self.feature[node]
                observed = row[feature_index]
                go_left = (
                    self.missing_go_to_left[node]
                    if math.isnan(observed)
                    else observed <= self.threshold[node]
                )
                node = (
                    self.children_left[node] if go_left else self.children_right[node]
                )
            predictions[row_index] = self.value[node]
        return predictions


@dataclass(frozen=True)
class LambdaMARTModel:
    candidate_id: str
    feature_names: tuple[str, ...]
    trees: tuple[RegressionTree, ...]
    learning_rate: float
    best_iteration: int
    training_groups: int
    training_rows: int
    seed: int

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> LambdaMARTModel:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        value["feature_names"] = tuple(value["feature_names"])
        value["trees"] = tuple(
            RegressionTree(
                **{
                    key: tuple(item[key])
                    for key in (
                        "children_left",
                        "children_right",
                        "feature",
                        "threshold",
                        "missing_go_to_left",
                        "value",
                    )
                }
            )
            for item in value["trees"]
        )
        return cls(**value)

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        scores: NDArray[np.float64] = np.zeros(len(features), dtype=np.float64)
        for tree in self.trees[: self.best_iteration]:
            scores += self.learning_rate * tree.predict(features)
        return scores

    def split_importance(self) -> dict[str, int]:
        counts = {name: 0 for name in self.feature_names}
        for tree in self.trees[: self.best_iteration]:
            for index in tree.feature:
                if index >= 0:
                    counts[self.feature_names[index]] += 1
        return counts


class LambdaMARTReranker:
    def __init__(self, features: GBDTFeatureStore, model: LambdaMARTModel) -> None:
        self.features = features
        self.model = model

    def rerank(
        self, query: str, ranking: list[str], *, rerank_k: int = 50
    ) -> list[str]:
        head = ranking[:rerank_k]
        if len(head) < 2:
            return list(ranking)
        matrix = self.features.matrix(query, head, self.model.feature_names)
        predictions = self.model.predict(matrix)
        original_ranks = {identifier: rank for rank, identifier in enumerate(head)}
        scores = {
            identifier: float(score)
            for identifier, score in zip(head, predictions, strict=True)
        }
        ordered = sorted(
            head,
            key=lambda identifier: (
                -scores[identifier],
                original_ranks[identifier],
                identifier,
            ),
        )
        return [*ordered, *ranking[rerank_k:]]


def fit_lambdamart(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: list[int],
    *,
    candidate_id: str,
    feature_names: tuple[str, ...],
    max_depth: int,
    num_leaves: int,
    learning_rate: float,
    max_rounds: int,
    early_stopping_rounds: int,
    validation: tuple[NDArray[np.float64], NDArray[np.int64], list[int]] | None = None,
    seed: int = 20260826,
) -> LambdaMARTModel:
    from sklearn.tree import DecisionTreeRegressor  # type: ignore[import-untyped]

    if len(features) != len(labels) or sum(groups) != len(labels):
        raise ValueError("ranking rows, labels, and group sizes do not align")
    if not groups or any(size < 2 for size in groups):
        raise ValueError("LambdaMART needs ranking groups with at least two rows")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature matrix width does not match feature names")
    if max_depth <= 0 or not 1 < num_leaves <= 2**max_depth:
        raise ValueError("invalid constrained tree capacity")
    group_slices = _group_slices(groups)
    train_scores: NDArray[np.float64] = np.zeros(len(labels), dtype=np.float64)
    valid_features: NDArray[np.float64] | None = None
    valid_labels: NDArray[np.int64] | None = None
    valid_groups: list[int] | None = None
    valid_scores: NDArray[np.float64] | None = None
    if validation is not None:
        valid_features, valid_labels, valid_groups = validation
        if len(valid_features) != len(valid_labels) or sum(valid_groups) != len(
            valid_labels
        ):
            raise ValueError("validation ranking groups do not align")
        valid_scores = np.zeros(len(valid_labels), dtype=np.float64)
    trees: list[RegressionTree] = []
    best_iteration = 0
    best_validation = -math.inf
    stale_rounds = 0
    for iteration in range(1, max_rounds + 1):
        lambdas, hessians = _ranking_derivatives(
            labels, train_scores, group_slices, truncation=10
        )
        estimator = DecisionTreeRegressor(
            criterion="squared_error",
            splitter="best",
            max_depth=max_depth,
            min_samples_leaf=40,
            min_impurity_decrease=1e-7,
            max_leaf_nodes=num_leaves,
            random_state=seed,
        )
        estimator.fit(features, lambdas, sample_weight=np.maximum(hessians, 1e-9))
        leaf_ids = estimator.apply(features)
        leaf_values: dict[int, float] = {}
        for leaf_id in np.unique(leaf_ids):
            selected = leaf_ids == leaf_id
            leaf_values[int(leaf_id)] = float(
                lambdas[selected].sum() / (hessians[selected].sum() + 1.0)
            )
        tree = _extract_tree(estimator.tree_, leaf_values)
        trees.append(tree)
        train_scores += learning_rate * tree.predict(features)
        if validation is None:
            best_iteration = iteration
            continue
        assert valid_features is not None
        assert valid_labels is not None
        assert valid_groups is not None
        assert valid_scores is not None
        valid_scores += learning_rate * tree.predict(valid_features)
        validation_ndcg = _mean_ndcg(valid_labels, valid_scores, valid_groups, k=10)
        if validation_ndcg > best_validation + 1e-12:
            best_validation = validation_ndcg
            best_iteration = iteration
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= early_stopping_rounds:
                break
    if best_iteration <= 0:
        best_iteration = len(trees)
    selected_trees = tuple(trees[:best_iteration])
    return LambdaMARTModel(
        candidate_id=candidate_id,
        feature_names=feature_names,
        trees=selected_trees,
        learning_rate=learning_rate,
        best_iteration=best_iteration,
        training_groups=len(groups),
        training_rows=len(labels),
        seed=seed,
    )


def _group_slices(groups: list[int]) -> list[slice]:
    result: list[slice] = []
    start = 0
    for size in groups:
        result.append(slice(start, start + size))
        start += size
    return result


def _ranking_derivatives(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    groups: list[slice],
    *,
    truncation: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lambdas: NDArray[np.float64] = np.zeros(len(labels), dtype=np.float64)
    hessians: NDArray[np.float64] = np.zeros(len(labels), dtype=np.float64)
    for group in groups:
        group_labels = labels[group]
        positives = np.flatnonzero(group_labels > 0)
        if len(positives) != 1:
            raise ValueError("each ranking group must contain exactly one positive")
        group_scores = scores[group]
        order = np.argsort(-group_scores, kind="stable")
        positions: NDArray[np.int64] = np.empty(len(order), dtype=np.int64)
        positions[order] = np.arange(len(order))
        positive = int(positives[0])
        positive_discount = (
            1.0 / math.log2(int(positions[positive]) + 2)
            if positions[positive] < truncation
            else 0.0
        )
        for raw_negative in np.flatnonzero(group_labels == 0):
            negative = int(raw_negative)
            negative_discount = (
                1.0 / math.log2(int(positions[negative]) + 2)
                if positions[negative] < truncation
                else 0.0
            )
            delta_ndcg = abs(positive_discount - negative_discount)
            if delta_ndcg == 0.0:
                continue
            margin = float(
                np.clip(group_scores[positive] - group_scores[negative], -40, 40)
            )
            probability = 1.0 / (1.0 + math.exp(margin))
            gradient = probability * delta_ndcg
            hessian = probability * (1.0 - probability) * delta_ndcg
            positive_index = group.start + positive
            negative_index = group.start + negative
            lambdas[positive_index] += gradient
            lambdas[negative_index] -= gradient
            hessians[positive_index] += hessian
            hessians[negative_index] += hessian
    return lambdas, hessians


def _extract_tree(tree: Any, leaf_values: dict[int, float]) -> RegressionTree:
    children_left = tuple(int(value) for value in tree.children_left)
    children_right = tuple(int(value) for value in tree.children_right)
    feature = tuple(int(value) for value in tree.feature)
    threshold = tuple(float(value) for value in tree.threshold)
    missing_raw = getattr(tree, "missing_go_to_left", np.zeros(tree.node_count))
    missing_go_to_left = tuple(bool(value) for value in missing_raw)
    values = tuple(leaf_values.get(index, 0.0) for index in range(tree.node_count))
    return RegressionTree(
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        missing_go_to_left=missing_go_to_left,
        value=values,
    )


def _mean_ndcg(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    groups: list[int],
    *,
    k: int,
) -> float:
    values = []
    for group in _group_slices(groups):
        order = np.argsort(-scores[group], kind="stable")[:k]
        ranked_labels = labels[group][order]
        positives = np.flatnonzero(ranked_labels > 0)
        values.append(
            0.0 if len(positives) == 0 else 1.0 / math.log2(int(positives[0]) + 2)
        )
    return float(np.mean(values))

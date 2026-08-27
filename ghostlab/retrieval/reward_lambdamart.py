from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ghostlab.evaluation.reward_deltas import (
    swap_reward_delta,
    terminal_session_reward,
)
from ghostlab.retrieval.gbdt import (
    LambdaMARTModel,
    RegressionTree,
    _extract_tree,
    _group_slices,
)

RewardObjective = Literal[
    "uniform_pairwise", "reward", "turn_aware_reward", "pointwise"
]

TECHNIQUE_IDS: dict[RewardObjective, str] = {
    "uniform_pairwise": "ranking.uniform_pairwise.v1",
    "reward": "ranking.reward_lambdamart.v1",
    "turn_aware_reward": "ranking.turn_aware_lambdamart.v1",
    "pointwise": "ranking.pointwise_gbdt.v1",
}


def fit_reward_lambdamart(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: list[int],
    turns: list[int],
    *,
    candidate_id: str,
    feature_names: tuple[str, ...],
    objective: RewardObjective,
    max_depth: int,
    num_leaves: int,
    learning_rate: float,
    max_rounds: int,
    early_stopping_rounds: int,
    validation: tuple[
        NDArray[np.float64], NDArray[np.int64], list[int], list[int]
    ]
    | None = None,
    min_samples_leaf: int = 40,
    l2_leaf: float = 1.0,
    seed: int = 20260826,
) -> LambdaMARTModel:
    """Fit a compact tree ranker with an organizer-reward-aware objective.

    ``turns`` are training labels used only to weight swaps. Runtime inference is
    identical to the existing target-free ``LambdaMARTModel.predict`` path.
    """
    from sklearn.tree import DecisionTreeRegressor  # type: ignore[import-untyped]

    _validate_dataset(features, labels, groups, turns, feature_names)
    if max_depth <= 0 or not 1 < num_leaves <= 2**max_depth:
        raise ValueError("invalid constrained tree capacity")
    if min_samples_leaf <= 0 or l2_leaf < 0.0:
        raise ValueError("invalid regularization")
    group_slices = _group_slices(groups)
    train_scores: NDArray[np.float64] = np.zeros(len(labels), dtype=np.float64)
    valid_features: NDArray[np.float64] | None = None
    valid_labels: NDArray[np.int64] | None = None
    valid_groups: list[int] | None = None
    valid_turns: list[int] | None = None
    valid_scores: NDArray[np.float64] | None = None
    if validation is not None:
        valid_features, valid_labels, valid_groups, valid_turns = validation
        _validate_dataset(
            valid_features, valid_labels, valid_groups, valid_turns, feature_names
        )
        valid_scores = np.zeros(len(valid_labels), dtype=np.float64)

    trees: list[RegressionTree] = []
    best_iteration = 0
    best_validation = -math.inf
    stale_rounds = 0
    for iteration in range(1, max_rounds + 1):
        lambdas, hessians = ranking_derivatives(
            labels, train_scores, group_slices, turns, objective=objective
        )
        estimator = DecisionTreeRegressor(
            criterion="squared_error",
            splitter="best",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
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
                lambdas[selected].sum() / (hessians[selected].sum() + l2_leaf)
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
        assert valid_turns is not None
        assert valid_scores is not None
        valid_scores += learning_rate * tree.predict(valid_features)
        metric = mean_predicted_terminal_reward(
            valid_labels, valid_scores, valid_groups, valid_turns
        )
        if metric > best_validation + 1e-12:
            best_validation = metric
            best_iteration = iteration
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= early_stopping_rounds:
                break
    if best_iteration <= 0:
        best_iteration = len(trees)
    return LambdaMARTModel(
        candidate_id=candidate_id,
        feature_names=feature_names,
        trees=tuple(trees[:best_iteration]),
        learning_rate=learning_rate,
        best_iteration=best_iteration,
        training_groups=len(groups),
        training_rows=len(labels),
        seed=seed,
    )


def ranking_derivatives(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    groups: list[slice],
    turns: list[int],
    *,
    objective: RewardObjective,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(groups) != len(turns):
        raise ValueError("one observable turn is required per ranking group")
    if objective == "pointwise":
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
        return labels.astype(np.float64) - probabilities, probabilities * (
            1.0 - probabilities
        )

    lambdas: NDArray[np.float64] = np.zeros(len(labels), dtype=np.float64)
    hessians: NDArray[np.float64] = np.zeros(len(labels), dtype=np.float64)
    for group, observed_turn in zip(groups, turns, strict=True):
        group_labels = labels[group]
        positives = np.flatnonzero(group_labels > 0)
        if len(positives) != 1:
            raise ValueError("each ranking group must contain exactly one positive")
        group_scores = scores[group]
        order = np.argsort(-group_scores, kind="stable")
        positions: NDArray[np.int64] = np.empty(len(order), dtype=np.int64)
        positions[order] = np.arange(len(order))
        positive = int(positives[0])
        for raw_negative in np.flatnonzero(group_labels == 0):
            negative = int(raw_negative)
            if objective == "uniform_pairwise":
                weight = 1.0
            else:
                turn = observed_turn if objective == "turn_aware_reward" else 1
                weight = swap_reward_delta(
                    int(positions[positive]) + 1,
                    int(positions[negative]) + 1,
                    turn,
                )
            if weight == 0.0:
                continue
            margin = float(
                np.clip(group_scores[positive] - group_scores[negative], -40, 40)
            )
            probability = 1.0 / (1.0 + math.exp(margin))
            gradient = probability * weight
            hessian = probability * (1.0 - probability) * weight
            positive_index = group.start + positive
            negative_index = group.start + negative
            lambdas[positive_index] += gradient
            lambdas[negative_index] -= gradient
            hessians[positive_index] += hessian
            hessians[negative_index] += hessian
    return lambdas, hessians


def mean_predicted_terminal_reward(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    groups: list[int],
    turns: list[int],
) -> float:
    if len(groups) != len(turns):
        raise ValueError("one observable turn is required per ranking group")
    values: list[float] = []
    for group, turn in zip(_group_slices(groups), turns, strict=True):
        order = np.argsort(-scores[group], kind="stable")
        ranked_labels = labels[group][order]
        positives = np.flatnonzero(ranked_labels > 0)
        if len(positives) != 1:
            raise ValueError("each ranking group must contain exactly one positive")
        rank = int(positives[0]) + 1
        values.append(terminal_session_reward(rank, turn))
    return float(np.mean(values))


def _validate_dataset(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: list[int],
    turns: list[int],
    feature_names: tuple[str, ...],
) -> None:
    if len(features) != len(labels) or sum(groups) != len(labels):
        raise ValueError("ranking rows, labels, and group sizes do not align")
    if not groups or any(size < 2 for size in groups):
        raise ValueError("LambdaMART needs ranking groups with at least two rows")
    if len(groups) != len(turns) or any(not 1 <= turn <= 10 for turn in turns):
        raise ValueError("observable turns must align with groups and be in 1..10")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature matrix width does not match feature names")

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from baseline.state import ASK_ORDER
from ghostlab.competition.contract import AskAttribute
from ghostlab.policy.signals import retrieval_signals
from ghostlab.retrieval.sparse import query_terms
from ghostlab.state.memory import ConversationState

QuestionAction: TypeAlias = AskAttribute | None
SPECIFIC_ACTIONS: tuple[AskAttribute, ...] = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
)
ACTION_ORDER: tuple[QuestionAction, ...] = (
    "other",
    "use_case",
    "feature",
    "material",
    "size",
    "style",
    "color",
    "budget",
    "brand",
    "category",
    None,
)
FEATURE_NAMES = (
    "turn_fraction",
    "turns_remaining_fraction",
    "query_term_fraction",
    "message_term_fraction",
    "active_slot_fraction",
    "asked_fraction",
    "no_preference_fraction",
    "current_turn_value_fraction",
    "simulator_answer_fraction",
    "response_no_preference",
    "previous_was_other",
    "candidate_count_fraction",
    "score_margin",
    "score_entropy",
    "score_concentration",
    "category_known",
    *tuple(f"active.{name}" for name in ASK_ORDER),
    *tuple(f"asked.{name}" for name in ASK_ORDER),
    *tuple(f"no_preference.{name}" for name in ASK_ORDER),
)


def legal_question_actions(state: ConversationState) -> tuple[QuestionAction, ...]:
    """Return official actions that can still reveal information, plus stop.

    ``other`` remains legal because it asks for an undisclosed preference rather
    than a typed slot. Specific questions are asked at most once and are removed
    once known or explicitly declined. Stop (``None``) is always legal.
    """

    active = {item.attribute for item in state.active_values()}
    unavailable = active | state.no_preference_attributes | set(state.asked_attributes)
    actions: list[QuestionAction] = ["other"]
    actions.extend(action for action in SPECIFIC_ACTIONS if action not in unavailable)
    actions.append(None)
    return tuple(actions)


def observable_question_features(
    state: ConversationState,
    *,
    message: str,
    query: str,
    turn: int,
    retrieval_scores: list[float],
) -> dict[str, float]:
    """Build target-free features available immediately before the action."""

    active_values = state.active_values()
    active = {item.attribute for item in active_values}
    current_values = [item for item in active_values if item.source_turn == turn]
    simulator_values = [
        item for item in current_values if item.provenance == "simulator_answer"
    ]
    scores = retrieval_signals(retrieval_scores)
    if retrieval_scores:
        minimum = min(retrieval_scores)
        nonnegative = [score - minimum + 1e-12 for score in retrieval_scores]
        total = sum(nonnegative)
        concentration = sum(nonnegative[:10]) / total if total else 0.0
        scale = abs(retrieval_scores[0]) + 1e-9
        margin = (
            0.0 if scores.top1_margin is None else math.tanh(scores.top1_margin / scale)
        )
    else:
        concentration = margin = 0.0
    values: dict[str, float] = {
        "turn_fraction": turn / 10.0,
        "turns_remaining_fraction": (10 - turn) / 9.0,
        "query_term_fraction": min(1.0, len(query_terms(query)) / 40.0),
        "message_term_fraction": min(1.0, len(query_terms(message)) / 20.0),
        "active_slot_fraction": min(1.0, len(active_values) / 8.0),
        "asked_fraction": min(1.0, len(state.asked_attributes) / 8.0),
        "no_preference_fraction": min(1.0, len(state.no_preference_attributes) / 8.0),
        "current_turn_value_fraction": min(1.0, len(current_values) / 4.0),
        "simulator_answer_fraction": min(1.0, len(simulator_values) / 4.0),
        "response_no_preference": float("don't have" in message.casefold()),
        "previous_was_other": float(state.last_asked_attribute == "other"),
        "candidate_count_fraction": min(1.0, scores.candidate_count / 200.0),
        "score_margin": margin,
        "score_entropy": 0.0
        if scores.normalized_entropy is None
        else scores.normalized_entropy,
        "score_concentration": concentration,
        "category_known": float("category" in active),
    }
    asked = set(state.asked_attributes)
    for attribute in ASK_ORDER:
        values[f"active.{attribute}"] = float(attribute in active)
    for attribute in ASK_ORDER:
        values[f"asked.{attribute}"] = float(attribute in asked)
    for attribute in ASK_ORDER:
        values[f"no_preference.{attribute}"] = float(
            attribute in state.no_preference_attributes
        )
    if tuple(values) != FEATURE_NAMES:
        raise RuntimeError("observable question feature schema drifted")
    return values


@dataclass(frozen=True)
class QuestionTrainingState:
    sample_id: str
    turn: int
    features: dict[str, float]
    action_rewards: dict[QuestionAction, float]


@dataclass(frozen=True)
class LinearActionValueModel:
    feature_names: tuple[str, ...]
    action_weights: dict[QuestionAction, tuple[float, ...]]
    l2: float
    training_states: int

    def __post_init__(self) -> None:
        expected = len(self.feature_names) + 1
        if set(self.action_weights) != set(ACTION_ORDER):
            raise ValueError("linear question model has an incomplete action set")
        if any(len(weights) != expected for weights in self.action_weights.values()):
            raise ValueError("linear question model feature count mismatch")

    def values(
        self,
        features: dict[str, float],
        legal_actions: tuple[QuestionAction, ...],
    ) -> dict[QuestionAction, float]:
        unknown = set(features) ^ set(self.feature_names)
        if unknown:
            raise ValueError(f"question feature mismatch: {sorted(unknown)}")
        vector: NDArray[np.float64] = np.asarray(
            [1.0, *(features[name] for name in self.feature_names)], dtype=np.float64
        )
        return {
            action: float(vector @ np.asarray(self.action_weights[action]))
            for action in legal_actions
        }

    def decide(
        self,
        features: dict[str, float],
        legal_actions: tuple[QuestionAction, ...],
    ) -> tuple[QuestionAction, dict[QuestionAction, float]]:
        if not legal_actions:
            raise ValueError("at least one legal question action is required")
        values = self.values(features, legal_actions)
        action = min(
            legal_actions,
            key=lambda item: (-values[item], ACTION_ORDER.index(item)),
        )
        return action, values


def fit_linear_action_value(
    states: list[QuestionTrainingState], *, l2: float = 1.0
) -> LinearActionValueModel:
    """Fit one regularized linear reward surface per action.

    The caller is responsible for passing training-fold sessions only. The dense
    counterfactual table means each action is learned from every state where it was
    legal, while the session remains the grouping unit for outer validation.
    """

    if not states or l2 < 0.0:
        raise ValueError("question training states and non-negative l2 are required")
    weights: dict[QuestionAction, tuple[float, ...]] = {}
    width = len(FEATURE_NAMES) + 1
    penalty: NDArray[np.float64] = np.eye(width, dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for action in ACTION_ORDER:
        eligible = [state for state in states if action in state.action_rewards]
        if not eligible:
            missing = np.zeros(width, dtype=np.float64)
            missing[0] = -1e6
            weights[action] = tuple(float(value) for value in missing)
            continue
        matrix: NDArray[np.float64] = np.asarray(
            [
                [1.0, *(state.features[name] for name in FEATURE_NAMES)]
                for state in eligible
            ],
            dtype=np.float64,
        )
        target: NDArray[np.float64] = np.asarray(
            [state.action_rewards[action] for state in eligible], dtype=np.float64
        )
        solution = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)
        weights[action] = tuple(float(value) for value in solution)
    return LinearActionValueModel(FEATURE_NAMES, weights, l2, len(states))

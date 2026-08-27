from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ghostlab.competition.contract import AskAttribute
from ghostlab.policy.models import JointAction
from ghostlab.policy.signals import retrieval_signals
from ghostlab.state.memory import ConversationState

JOINT_FEATURE_NAMES = (
    "turn_fraction",
    "active_slot_fraction",
    "asked_fraction",
    "no_preference_fraction",
    "previous_candidate_fraction",
    "previous_margin",
    "previous_entropy",
    "category_known",
)


def observable_joint_features(
    state: ConversationState,
    *,
    turn: int,
    previous_scores: list[float] | None,
) -> dict[str, float]:
    signals = retrieval_signals(previous_scores or ())
    if previous_scores:
        scale = abs(previous_scores[0]) + 1e-9
        margin = (
            0.0
            if signals.top1_margin is None
            else math.tanh(signals.top1_margin / scale)
        )
    else:
        margin = 0.0
    active = {item.attribute for item in state.active_values()}
    values = {
        "turn_fraction": turn / 10.0,
        "active_slot_fraction": min(1.0, len(active) / 8.0),
        "asked_fraction": min(1.0, len(state.asked_attributes) / 8.0),
        "no_preference_fraction": min(
            1.0, len(state.no_preference_attributes) / 8.0
        ),
        "previous_candidate_fraction": min(1.0, signals.candidate_count / 200.0),
        "previous_margin": margin,
        "previous_entropy": 0.0
        if signals.normalized_entropy is None
        else signals.normalized_entropy,
        "category_known": float("category" in active),
    }
    if tuple(values) != JOINT_FEATURE_NAMES:
        raise RuntimeError("observable joint feature schema drifted")
    return values


def legal_question_attributes(state: ConversationState) -> frozenset[AskAttribute | None]:
    unavailable = (
        {item.attribute for item in state.active_values()}
        | set(state.asked_attributes)
        | state.no_preference_attributes
    )
    allowed: set[AskAttribute | None] = {
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
        None,
    }
    return frozenset(
        action for action in allowed if action is None or action not in unavailable
    )


def legalize_joint_action(
    action: JointAction,
    state: ConversationState,
    *,
    allowed_routes: frozenset[str],
    allowed_depths: frozenset[int],
    base_action: JointAction,
) -> JointAction:
    """Fail closed to registered route/depth and stop an illegal question."""

    route = (
        action.retrieval_route
        if action.retrieval_route in allowed_routes
        else base_action.retrieval_route
    )
    depth = (
        action.retrieval_k
        if action.retrieval_k in allowed_depths
        else base_action.retrieval_k
    )
    question = (
        action.ask_attribute
        if action.ask_attribute in legal_question_attributes(state)
        else None
    )
    return action.model_copy(
        update={
            "ask_attribute": question,
            "retrieval_route": route,
            "retrieval_k": depth,
        }
    )


@dataclass(frozen=True)
class JointTrainingState:
    sample_id: str
    turn: int
    features: Mapping[str, float]
    action_rewards: Mapping[str, float]

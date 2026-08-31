from __future__ import annotations

import pytest

from ghostlab.retrieval.residual import (
    ResidualAgentAdapter,
    ResidualDecision,
    ResidualPolicy,
    membership_preserving_reorder,
)
from ghostlab.state.memory import ConversationState


class _ParentAgent:
    def __init__(self, recommendations: list[object]) -> None:
        self.recommendations = recommendations
        self.sessions = {"session": ConversationState("session", {})}
        self.last_runtime_inputs = {"session": ("shoe", [1.0, 0.5])}
        self.retrieval_trace: list[dict[str, object]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        del session_id, user_profile

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        del session_id, user_message, turn, top_k
        return {
            "message": "matches",
            "ask_attribute": None,
            "recommendations": self.recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


class _ReversingResidual:
    def __init__(self, *, activate: bool = True) -> None:
        self.activate = activate
        self.observed: tuple[str, ...] | None = None

    def rerank(
        self, query: str, ranking: tuple[str, ...], **kwargs: object
    ) -> ResidualDecision:
        del query, kwargs
        self.observed = ranking
        reordered = tuple(reversed(ranking)) if self.activate else ranking
        return ResidualDecision(
            reordered,
            self.activate,
            "activated" if self.activate else "confidence_gate",
            0.1 if self.activate else 0.0,
            len(ranking) if self.activate else 0,
        )


def test_reorders_without_changing_membership() -> None:
    original = ["A", "B", "C", "D"]
    result = membership_preserving_reorder(
        original,
        [0.1, 0.8, 0.2, 0.05],
        ResidualPolicy(rerank_depth=3, maximum_moved_ids=3),
    )
    assert result.activated
    assert result.ranking == ("B", "C", "A", "D")
    assert set(result.ranking) == set(original)


def test_confidence_gate_preserves_original_exactly() -> None:
    original = ["A", "B", "C"]
    result = membership_preserving_reorder(
        original,
        [0.4, 0.41, 0.39],
        ResidualPolicy(minimum_probability_margin=0.05),
    )
    assert not result.activated
    assert result.reason == "probability_margin"
    assert result.ranking == tuple(original)


def test_movement_gate_preserves_original_exactly() -> None:
    original = ["A", "B", "C", "D"]
    result = membership_preserving_reorder(
        original,
        [0.1, 0.2, 0.3, 0.9],
        ResidualPolicy(maximum_moved_ids=2),
    )
    assert not result.activated
    assert result.reason == "movement_limit"
    assert result.ranking == tuple(original)


def test_duplicate_membership_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        membership_preserving_reorder(["A", "A"], [0.1, 0.2], ResidualPolicy())


def test_target_presence_is_invariant_for_every_possible_target() -> None:
    original = ["A", "B", "C", "D"]
    result = membership_preserving_reorder(
        original,
        [0.1, 0.8, 0.2, 0.05],
        ResidualPolicy(rerank_depth=4, maximum_moved_ids=4),
    )
    for target in [*original, "ABSENT"]:
        assert (target in result.ranking) == (target in original)


def test_runtime_adapter_extracts_ids_and_preserves_response_objects() -> None:
    recommendations = [
        {"parent_asin": "A", "score": 0.9},
        {"parent_asin": "B", "score": 0.8},
    ]
    residual = _ReversingResidual()
    response = ResidualAgentAdapter(
        _ParentAgent(recommendations),
        residual,  # type: ignore[arg-type]
    ).respond("session", "shoe", 1, 10)

    assert residual.observed == ("A", "B")
    assert response["recommendations"] == [recommendations[1], recommendations[0]]
    assert all(isinstance(item, dict) for item in response["recommendations"])


def test_runtime_adapter_is_exactly_inert_when_gate_does_not_activate() -> None:
    recommendations = [{"parent_asin": "A"}, {"parent_asin": "B"}]
    parent = _ParentAgent(recommendations)
    response = ResidualAgentAdapter(
        parent,
        _ReversingResidual(activate=False),  # type: ignore[arg-type]
    ).respond("session", "shoe", 1, 10)

    assert response["recommendations"] is recommendations

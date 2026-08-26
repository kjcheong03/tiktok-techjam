"""Deterministic question policies used by baseline evaluations.

Question selection is intentionally kept separate from conversation state.  A
policy receives the state object and may return the attribute to ask about;
the current-order policy delegates to the existing state method so its
semantics stay backward compatible.  The fixed-``other`` policy is a
diagnostic probe and does not inspect or mutate state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .state import fixed_question_for_turn


class QuestionState(Protocol):
    """The small state surface needed by a question policy."""

    def choose_question(self) -> str | None:
        ...


QuestionPolicy = Callable[[QuestionState, int | None], str | None]


def current_order(state: QuestionState, turn: int | None = None) -> str | None:
    """Return the next attribute according to the existing state semantics.

    ``turn`` is accepted for a common policy call signature, but the current
    V1 order is driven entirely by ``SessionState.choose_question``.  Keeping
    that call in one place means V1 policy evaluation cannot accidentally
    diverge from the production baseline.
    """

    del turn
    return state.choose_question()


def fixed_turn_order(state: QuestionState, turn: int | None = None) -> str | None:
    """Return the original fixed turn order without reading managed state."""

    del state
    if not isinstance(turn, int):
        raise ValueError("fixed_turn_order requires a concrete integer turn")
    return fixed_question_for_turn(turn)


def fixed_other(state: QuestionState, turn: int | None = None) -> str:
    """Return the simulator-sensitive fixed-``other`` diagnostic probe."""

    del state, turn
    return "other"


QUESTION_POLICIES: dict[str, QuestionPolicy] = {
    "current_order": current_order,
    "fixed_turn_order": fixed_turn_order,
    "fixed_other": fixed_other,
}


def get_question_policy(name: str) -> QuestionPolicy:
    """Resolve a named deterministic question policy.

    A ``ValueError`` gives command-line callers a useful error while keeping
    the registry extensible for later non-adaptive experiments.
    """

    try:
        return QUESTION_POLICIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(QUESTION_POLICIES))
        raise ValueError(f"unknown question policy {name!r}; choose one of: {available}") from exc


__all__ = [
    "QuestionPolicy",
    "QuestionState",
    "QUESTION_POLICIES",
    "current_order",
    "fixed_turn_order",
    "fixed_other",
    "get_question_policy",
]

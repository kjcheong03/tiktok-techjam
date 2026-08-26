from __future__ import annotations

import re
from typing import Literal

from baseline.state import OVERRIDE_RE
from ghostlab.state.memory import ConversationState

QueryVariant = Literal[
    "raw_history",
    "structured_active",
    "category_constraints",
    "raw_plus_active",
    "compressed_raw",
    "negation_safe_hybrid",
]
NEGATED_CLAUSE_RE = re.compile(
    r"\b(?:not|avoid|without|exclude)\b[^.;,]*", re.IGNORECASE
)


def _unique(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        cleaned = " ".join(part.split()).strip(" .;,\t\n")
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _active_parts(state: ConversationState, *, include_other: bool = True) -> list[str]:
    override_turn = max(
        (
            turn
            for turn, message in enumerate(state.messages, start=1)
            if OVERRIDE_RE.search(message)
        ),
        default=0,
    )
    values = sorted(
        (
            item
            for item in state.active_values()
            if not override_turn
            or item.attribute == "category"
            or item.source_turn >= override_turn
        ),
        key=lambda item: (
            item.attribute != "category",
            item.source_turn,
            item.normalized,
        ),
    )
    return _unique(
        [item.value for item in values if include_other or item.attribute != "other"]
    )


def _compressed_messages(state: ConversationState) -> list[str]:
    messages = (
        state.messages
        if len(state.messages) <= 4
        else [state.messages[0], *state.messages[-3:]]
    )
    return _unique([" ".join(message.split()[:24]) for message in messages])


def build_query(state: ConversationState, variant: QueryVariant) -> str:
    """Build a non-destructive query from runtime-observable conversation state."""

    raw = _unique(state.messages)
    active = _active_parts(state)
    if variant == "raw_history":
        parts = raw
    elif variant == "structured_active":
        parts = active
    elif variant == "category_constraints":
        parts = _active_parts(state, include_other=False)
    elif variant == "raw_plus_active":
        parts = [*active, *raw]
    elif variant == "compressed_raw":
        parts = _compressed_messages(state)
    else:
        latest = (
            NEGATED_CLAUSE_RE.sub(" ", state.messages[-1]) if state.messages else ""
        )
        parts = [*active, latest]
    fallback = state.messages[-1] if state.messages else ""
    return ". ".join(_unique(parts)) or fallback

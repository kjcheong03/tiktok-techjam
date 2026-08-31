from __future__ import annotations

import re
from typing import Literal

from ghostlab.state.memory import ConversationState, MemoryValue

DenseQueryVariant = Literal[
    "raw_history", "raw_plus_active", "negation_safe_structured"
]
QUERY_VARIANTS: tuple[DenseQueryVariant, ...] = (
    "raw_history",
    "raw_plus_active",
    "negation_safe_structured",
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .;,")


def _safe_positive_values(state: ConversationState) -> list[MemoryValue]:
    negatives = [item.normalized for item in state.active_values("negative")]
    values = []
    for item in state.active_values():
        if item.attribute in state.no_preference_attributes:
            continue
        if any(
            negative in item.normalized or item.normalized in negative
            for negative in negatives
        ):
            continue
        values.append(item)
    return sorted(
        values,
        key=lambda item: (
            item.attribute != "category",
            item.source_turn,
            item.attribute,
            item.normalized,
        ),
    )


def active_structured_query(state: ConversationState) -> str:
    fields = []
    seen: set[tuple[str, str]] = set()
    for item in _safe_positive_values(state):
        key = item.attribute, item.normalized
        if key in seen:
            continue
        seen.add(key)
        fields.append(
            f"{item.attribute.replace('_', ' ').title()}: {_clean(item.value)}"
        )
    return ". ".join(fields)


def build_dense_query(state: ConversationState, variant: DenseQueryVariant) -> str:
    raw = ". ".join(state.messages)
    if variant == "raw_history":
        return raw
    structured = active_structured_query(state)
    if variant == "raw_plus_active":
        return (
            raw
            if not structured
            else f"Raw history: {raw}. Active preferences: {structured}"
        )
    if variant == "negation_safe_structured":
        return structured or state.messages[-1]
    raise ValueError(f"unknown dense query variant: {variant}")

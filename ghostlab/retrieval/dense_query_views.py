"""Deterministic State V2 query views for diverse dense retrieval."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ghostlab.state.baseline_v2 import StateBaselineV2, StructuredConstraint
from ghostlab.state.v2_view import ConstraintView, V2StateView

DenseQueryViewName = Literal[
    "complete_request", "use_case", "features_style", "profile_context"
]
ConstraintLike = StructuredConstraint | ConstraintView

_USE_CASE_ATTRIBUTES = ("occasion", "use_case")
_FEATURE_STYLE_ATTRIBUTES = ("feature", "style", "material", "other")


@dataclass(frozen=True)
class DenseQueryView:
    name: DenseQueryViewName
    query_text: str


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" .;,\t\n")


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _is_active(constraint: ConstraintLike) -> bool:
    return not isinstance(constraint, StructuredConstraint) or constraint.active


def _positive_values(
    constraints: Sequence[ConstraintLike],
) -> dict[str, list[str]]:
    negative_values = _unique(
        [
            value
            for constraint in constraints
            if _is_active(constraint) and constraint.polarity == "exclude"
            for value in constraint.values
        ]
    )
    negative_keys = [value.casefold() for value in negative_values]
    result: dict[str, list[str]] = {}
    for constraint in constraints:
        if not _is_active(constraint) or constraint.polarity != "include":
            continue
        for value in constraint.values:
            cleaned = _clean(value)
            key = cleaned.casefold()
            if not cleaned or any(
                negative in key or key in negative for negative in negative_keys
            ):
                continue
            bucket = result.setdefault(constraint.attribute, [])
            if key not in {item.casefold() for item in bucket}:
                bucket.append(cleaned)
    return result


def _group_query(positive: dict[str, list[str]], attributes: Sequence[str]) -> str:
    group_values = _unique(
        [value for attribute in attributes for value in positive.get(attribute, [])]
    )
    if not group_values:
        return ""
    return ". ".join(_unique([*positive.get("category", []), *group_values]))


def build_dense_query_views(
    state: StateBaselineV2 | V2StateView | None = None,
    *,
    current_request: str | None = None,
    active_constraints: Sequence[ConstraintLike] | None = None,
) -> tuple[DenseQueryView, ...]:
    """Build deduplicated complete, use-case and feature/style query views."""
    if state is not None and not isinstance(state, (StateBaselineV2, V2StateView)):
        raise TypeError("state must be StateBaselineV2 or V2StateView")
    if current_request is None:
        if isinstance(state, V2StateView):
            current_request = state.query_text
        elif isinstance(state, StateBaselineV2):
            current_request = ". ".join(state.messages)
        else:
            raise ValueError("current_request is required without State V2 input")
    if active_constraints is None:
        active_constraints = (
            state.active_constraints
            if isinstance(state, (StateBaselineV2, V2StateView))
            else ()
        )

    positive = _positive_values(active_constraints)
    candidates: tuple[tuple[DenseQueryViewName, str], ...] = (
        ("complete_request", current_request),
        ("use_case", _group_query(positive, _USE_CASE_ATTRIBUTES)),
        ("features_style", _group_query(positive, _FEATURE_STYLE_ATTRIBUTES)),
    )
    views: list[DenseQueryView] = []
    seen: set[str] = set()
    for name, query_text in candidates:
        cleaned = _clean(query_text)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            views.append(DenseQueryView(name=name, query_text=cleaned))
    return tuple(views)


__all__ = ["DenseQueryView", "DenseQueryViewName", "build_dense_query_views"]

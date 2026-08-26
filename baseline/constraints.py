"""Small, typed constraint contract used by the State Baseline V2.

The V2 state deliberately keeps interpretation separate from state transitions.  The
adapter in this module delegates extraction to :mod:`baseline.state`, while callers
that have a newer interpreter can construct :class:`StructuredConstraint` values
directly against the same contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

from .state import SessionState


ALLOWED_ATTRIBUTES = (
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
)
Attribute = Literal[
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
]
Relation = Literal["any", "all"]
Polarity = Literal["include", "exclude"]
Strength = Literal["hard", "soft", "unspecified"]
Operator = Literal["equals", "at_most", "at_least", "none"]
Provenance = Literal["explicit", "simulator_answer", "inferred"]
ConstraintStatus = Literal["active", "superseded"]


def normalize_value(value: object) -> str:
    """Return the stable, case-insensitive representation used for state keys."""

    return re.sub(r"\s+", " ", str(value)).strip(" .;,\t\n").casefold()


@dataclass
class StructuredConstraint:
    """A normalized piece of customer evidence.

    ``source_text`` is intentionally not normalized: retaining the exact customer
    message is part of the V2 contract.  Constraint values are normalized and kept
    in source order; a list is used instead of a set so a caller can express an
    ``any`` or ``all`` group deterministically.
    """

    attribute: Attribute
    values: list[str]
    relation: Relation = "any"
    polarity: Polarity = "include"
    strength: Strength = "unspecified"
    operator: Operator = "none"
    source_turn: int = 0
    source_text: str = ""
    provenance: Provenance = "explicit"
    status: ConstraintStatus = "active"

    def __post_init__(self) -> None:
        if self.attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError(f"unsupported constraint attribute: {self.attribute!r}")
        if self.relation not in {"any", "all"}:
            raise ValueError(f"unsupported constraint relation: {self.relation!r}")
        if self.polarity not in {"include", "exclude"}:
            raise ValueError(f"unsupported constraint polarity: {self.polarity!r}")
        if self.strength not in {"hard", "soft", "unspecified"}:
            raise ValueError(f"unsupported constraint strength: {self.strength!r}")
        if self.operator not in {"equals", "at_most", "at_least", "none"}:
            raise ValueError(f"unsupported constraint operator: {self.operator!r}")
        if self.provenance not in {"explicit", "simulator_answer", "inferred"}:
            raise ValueError(f"unsupported constraint provenance: {self.provenance!r}")
        if self.status not in {"active", "superseded"}:
            raise ValueError(f"unsupported constraint status: {self.status!r}")
        if not isinstance(self.source_turn, int) or self.source_turn < 0:
            raise ValueError("source_turn must be a non-negative integer")

        raw_values: Sequence[object]
        if isinstance(self.values, str):
            raw_values = [self.values]
        else:
            raw_values = self.values
        normalized: list[str] = []
        for value in raw_values:
            item = normalize_value(value)
            if item and item not in normalized:
                normalized.append(item)
        if not normalized:
            raise ValueError("a constraint must contain at least one value")
        self.values = normalized

    @property
    def active(self) -> bool:
        """Compatibility convenience for code that used V1 ``SlotValue.active``."""

        return self.status == "active"

    @active.setter
    def active(self, value: bool) -> None:
        self.status = "active" if value else "superseded"

    def supersede(self) -> None:
        self.status = "superseded"

    def normalized_key(self) -> tuple[object, ...]:
        """Return the evidence identity used to avoid exact duplicate groups."""

        return (
            self.attribute,
            tuple(self.values),
            self.relation,
            self.polarity,
            self.strength,
            self.operator,
        )


@dataclass(frozen=True)
class LegacyParseResult:
    """V1 extraction output plus the separate no-preference signal."""

    constraints: tuple[StructuredConstraint, ...]
    no_preference_attributes: frozenset[str]


class LegacyV1ConstraintAdapter:
    """Translate the existing V1 parser output into the V2 contract.

    A fresh V1 ``SessionState`` is used for each message.  That preserves the
    existing V1 extraction rules while leaving supersession and accumulation to the
    V2 state container.  In particular, a V1 intent-override side effect cannot erase
    unrelated V2 constraints before the state layer gets to apply its targeted rule.
    """

    def parse(
        self,
        message: str,
        turn: int,
        last_asked_attribute: str | None = None,
    ) -> list[StructuredConstraint]:
        result = self.parse_result(
            message,
            turn,
            last_asked_attribute=last_asked_attribute,
        )
        return list(result.constraints)

    def parse_result(
        self,
        message: str,
        turn: int,
        last_asked_attribute: str | None = None,
    ) -> LegacyParseResult:
        # This intentionally delegates all message recognition to V1.  Do not add
        # aliases or a second natural-language parser here: the state-only ablation
        # needs the old interpreter and the new transition semantics to be separable.
        legacy = SessionState("v2-legacy-adapter", {})
        legacy.last_asked_attribute = last_asked_attribute
        legacy.observe(message, turn)
        # V1 marks earlier values for one attribute inactive while parsing a
        # single message.  Since this adapter creates a fresh state per message,
        # those statuses are parser artifacts rather than user supersession.
        constraints = tuple(
            StructuredConstraint(
                attribute=slot.attribute,  # type: ignore[arg-type]
                values=[slot.value],
                source_turn=slot.source_turn,
                source_text=slot.source_text,
                provenance=slot.provenance,  # type: ignore[arg-type]
                status="active",
            )
            for slot in legacy.slots
        )
        no_preference = frozenset(legacy.no_preference_attributes)
        return LegacyParseResult(constraints, no_preference)

__all__ = [
    "ALLOWED_ATTRIBUTES",
    "Attribute",
    "Relation",
    "Polarity",
    "Strength",
    "Operator",
    "Provenance",
    "ConstraintStatus",
    "StructuredConstraint",
    "LegacyParseResult",
    "LegacyV1ConstraintAdapter",
    "normalize_value",
]

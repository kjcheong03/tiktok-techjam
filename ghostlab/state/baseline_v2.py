"""Native adapter for the teammate State Baseline V2.

The implementation preserves the teammate branch's deterministic constraint
transitions while presenting the :class:`ConversationState` interface used by
the unified runtime.  The legacy parser remains deliberately separate from the
state reducer so later interpreters can emit the same constraint contract.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from baseline.state import ASK_ORDER, SessionState
from ghostlab.state.memory import ConversationState, MemoryValue

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
ConstraintPolarity = Literal["include", "exclude"]
Strength = Literal["hard", "soft", "unspecified"]
Operator = Literal["equals", "at_most", "at_least", "none"]
ConstraintProvenance = Literal["explicit", "simulator_answer", "inferred"]
ConstraintStatus = Literal["active", "superseded"]

_LOW_COVERAGE_MAX_ACTIVE_CONSTRAINTS = 3
_CORRECTION_RE = re.compile(
    r"\b(?:actually|instead|ignore|replace|replacement|correction|rather)\b",
    re.IGNORECASE,
)
_VAGUE_CORRECTION_VALUES = {
    "something",
    "anything",
    "whatever",
    "nothing",
    "another",
    "different",
    "the best",
    "a good one",
    "the right one",
}


def normalize_value(value: object) -> str:
    """Return the stable case-insensitive representation used by state keys."""

    return re.sub(r"\s+", " ", str(value)).strip(" .;,\t\n").casefold()


@dataclass
class StructuredConstraint:
    """Typed evidence contract retained from the teammate baseline."""

    attribute: Attribute
    values: list[str]
    relation: Relation = "any"
    polarity: ConstraintPolarity = "include"
    strength: Strength = "unspecified"
    operator: Operator = "none"
    source_turn: int = 0
    source_text: str = ""
    provenance: ConstraintProvenance = "explicit"
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
        raw_values = [self.values] if isinstance(self.values, str) else self.values
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
        return self.status == "active"

    def supersede(self) -> None:
        self.status = "superseded"

    def normalized_key(self) -> tuple[object, ...]:
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
    constraints: tuple[StructuredConstraint, ...]
    no_preference_attributes: frozenset[str]


class LegacyConstraintAdapter:
    """Translate the frozen starter parser output into structured constraints."""

    def parse_result(
        self,
        message: str,
        turn: int,
        last_asked_attribute: str | None = None,
    ) -> LegacyParseResult:
        legacy = SessionState("state-baseline-v2-adapter", {})
        legacy.last_asked_attribute = last_asked_attribute
        legacy.observe(message, turn)
        constraints = tuple(
            StructuredConstraint(
                attribute=slot.attribute,  # type: ignore[arg-type]
                values=[slot.value],
                source_turn=slot.source_turn,
                source_text=slot.source_text,
                provenance=slot.provenance,  # type: ignore[arg-type]
            )
            for slot in legacy.slots
        )
        return LegacyParseResult(
            constraints=constraints,
            no_preference_attributes=frozenset(legacy.no_preference_attributes),
        )


@dataclass
class StateBaselineV2(ConversationState):
    """Structured V2 state with lossless audit evidence and unified compatibility."""

    intent_epoch: int = 0
    constraints: list[StructuredConstraint] = field(default_factory=list)
    _adapter: LegacyConstraintAdapter = field(
        default_factory=LegacyConstraintAdapter,
        repr=False,
        compare=False,
    )

    @property
    def active_constraints(self) -> list[StructuredConstraint]:
        return [constraint for constraint in self.constraints if constraint.active]

    def observe(
        self,
        message: str,
        turn: int,
        parsed_constraints: Iterable[StructuredConstraint] | None = None,
        *,
        no_preference_attributes: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if not isinstance(turn, int) or turn < 0:
            raise ValueError("turn must be a non-negative integer")
        self.messages.append(message)

        no_preference = {
            normalize_value(attribute)
            for attribute in (no_preference_attributes or ())
            if normalize_value(attribute)
        }
        if parsed_constraints is None:
            parsed = self._adapter.parse_result(
                message,
                turn,
                last_asked_attribute=self.last_asked_attribute,
            )
            incoming = list(parsed.constraints)
            no_preference.update(parsed.no_preference_attributes)
        else:
            incoming = list(parsed_constraints)

        self.no_preference_attributes.update(no_preference)
        correction = _CORRECTION_RE.search(message) is not None
        accepted = self.apply_constraints(
            incoming,
            source_text=message,
            correction=correction,
        )
        if correction and accepted:
            self.intent_epoch += 1

    def apply_constraints(
        self,
        constraints: Iterable[StructuredConstraint],
        *,
        source_text: str | None = None,
        correction: bool = False,
        supersede_attributes: Iterable[str] | None = None,
    ) -> list[StructuredConstraint]:
        incoming = [self._with_source_text(item, source_text) for item in constraints]
        if not incoming:
            return []

        replacement_attributes = {
            normalize_value(attribute)
            for attribute in (supersede_attributes or ())
            if normalize_value(attribute) in ALLOWED_ATTRIBUTES
        }
        if correction:
            replacement_attributes.update(
                constraint.attribute
                for constraint in incoming
                if self._confident_correction_attribute(constraint)
            )
        for attribute in replacement_attributes:
            self.supersede_attribute(attribute)

        accepted: list[StructuredConstraint] = []
        for constraint in incoming:
            if correction and not self._confident_correction_attribute(constraint):
                continue
            if constraint.attribute not in ALLOWED_ATTRIBUTES:
                continue
            if constraint.active:
                self.no_preference_attributes.discard(constraint.attribute)
            if self._is_duplicate_active(constraint):
                continue
            self.constraints.append(constraint)
            accepted.append(constraint)
        self._sync_memory_values()
        return accepted

    @staticmethod
    def _with_source_text(
        constraint: StructuredConstraint,
        source_text: str | None,
    ) -> StructuredConstraint:
        if not isinstance(constraint, StructuredConstraint):
            raise TypeError("constraints must contain StructuredConstraint values")
        if source_text is not None and not constraint.source_text:
            return replace(constraint, source_text=source_text)
        return constraint

    def _is_duplicate_active(self, candidate: StructuredConstraint) -> bool:
        key = candidate.normalized_key()
        return any(
            existing.active and existing.normalized_key() == key
            for existing in self.constraints
        )

    @staticmethod
    def _confident_correction_attribute(constraint: StructuredConstraint) -> bool:
        if constraint.attribute != "feature":
            return True
        value = " ".join(constraint.values).strip()
        if value in _VAGUE_CORRECTION_VALUES:
            return False
        if value.startswith(("something ", "anything ")):
            remainder = value.split(" ", 1)[1].strip()
            return remainder not in _VAGUE_CORRECTION_VALUES
        return True

    def supersede_attribute(self, attribute: str) -> int:
        normalized_attribute = normalize_value(attribute)
        if normalized_attribute not in ALLOWED_ATTRIBUTES:
            return 0
        changed = 0
        for constraint in self.constraints:
            if constraint.attribute == normalized_attribute and constraint.active:
                constraint.supersede()
                changed += 1
        self._sync_memory_values()
        return changed

    def constraint_values(
        self,
        attribute: str | None = None,
        *,
        polarity: ConstraintPolarity = "include",
    ) -> list[str]:
        normalized_attribute = (
            normalize_value(attribute) if attribute is not None else None
        )
        result: list[str] = []
        for constraint in self.constraints:
            if not constraint.active or constraint.polarity != polarity:
                continue
            if (
                normalized_attribute is not None
                and constraint.attribute != normalized_attribute
            ):
                continue
            for value in constraint.values:
                if value not in result:
                    result.append(value)
        return result

    def build_state_query(self) -> str:
        ordered = sorted(
            enumerate(self.constraints),
            key=lambda item: (
                item[1].attribute != "category",
                item[1].source_turn,
                item[0],
            ),
        )
        values: list[str] = []
        for _, constraint in ordered:
            if not constraint.active or constraint.polarity != "include":
                continue
            for value in constraint.values:
                if value not in values:
                    values.append(value)
        return ". ".join(values)

    def build_raw_history_query(self) -> str:
        return ". ".join(self.messages)

    def build_coverage_adaptive_query(self) -> str:
        state_query = self.build_state_query()
        raw_history = self.build_raw_history_query()
        has_superseded = any(not constraint.active for constraint in self.constraints)
        if (
            has_superseded
            and len(self.active_constraints) <= _LOW_COVERAGE_MAX_ACTIVE_CONSTRAINTS
        ):
            return raw_history
        return state_query or raw_history

    def build_query(self, compressed: bool = False) -> str:
        del compressed
        return self.build_state_query()

    def choose_question(
        self,
        order: tuple[str, ...] = ASK_ORDER,
        *,
        allow_other: bool = False,
    ) -> str | None:
        known = {constraint.attribute for constraint in self.active_constraints}
        unavailable = known | self.no_preference_attributes | set(self.asked_attributes)
        candidates = (*order, "other") if allow_other else order
        for attribute in candidates:
            if attribute in unavailable:
                continue
            self.asked_attributes.append(attribute)
            self.last_asked_attribute = attribute
            return attribute
        self.last_asked_attribute = None
        return None

    def reset(
        self,
        session_id: str | None = None,
        user_profile: dict | None = None,
    ) -> None:
        if session_id is not None:
            self.session_id = session_id
        if user_profile is not None:
            self.user_profile = user_profile
        self.messages.clear()
        self.values.clear()
        self.constraints.clear()
        self.asked_attributes.clear()
        self.no_preference_attributes.clear()
        self.last_asked_attribute = None
        self.intent_epoch = 0
        self._adapter = LegacyConstraintAdapter()

    def _sync_memory_values(self) -> None:
        self.values = [
            MemoryValue(
                attribute=constraint.attribute,
                value=value,
                normalized=normalize_value(value),
                source_turn=constraint.source_turn,
                source_text=constraint.source_text,
                provenance=(
                    constraint.provenance
                    if constraint.provenance != "inferred"
                    else "explicit"
                ),
                polarity=(
                    "positive" if constraint.polarity == "include" else "negative"
                ),
                active=constraint.active,
                invalidated_reason=(
                    None if constraint.active else "state_v2_superseded"
                ),
            )
            for constraint in self.constraints
            for value in constraint.values
        ]


__all__ = [
    "ALLOWED_ATTRIBUTES",
    "LegacyConstraintAdapter",
    "StateBaselineV2",
    "StructuredConstraint",
    "normalize_value",
]

"""State-only Baseline V2 transitions.

This module intentionally does not contain a new natural-language interpreter.  By
default, :class:`StructuredSessionState` obtains evidence through the V1 adapter in
``baseline.constraints``.  A future interpreter can provide the same structured
constraints directly to ``observe`` or ``apply_constraints``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterable

from .constraints import (
    ALLOWED_ATTRIBUTES,
    LegacyV1ConstraintAdapter,
    StructuredConstraint,
    normalize_value,
)
from .state import ASK_ORDER

if TYPE_CHECKING:
    from .interpreter import ParseResult


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


def _attribute_name(attribute: object) -> str:
    return normalize_value(attribute)


@dataclass
class StructuredSessionState:
    """A deterministic, replayable conversational state container.

    ``constraints`` is append-only evidence: a correction changes the status of
    prior records to ``superseded`` instead of deleting them.  This preserves the
    audit trail and makes state transitions measurable in transcript replay.
    """

    session_id: str
    user_profile: dict
    messages: list[str] = field(default_factory=list)
    constraints: list[StructuredConstraint] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    no_preference_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None
    interpreter: Callable[[str, int, str | None], "ParseResult"] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _adapter: LegacyV1ConstraintAdapter = field(
        default_factory=LegacyV1ConstraintAdapter,
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
        """Record a raw message and apply its structured evidence.

        With no explicit ``parsed_constraints`` argument the existing V1 parser is
        used through :class:`LegacyV1ConstraintAdapter`.  Passing constraints is the
        seam for a later interpreter and is useful for testing state transitions in
        isolation.
        """

        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if not isinstance(turn, int) or turn < 0:
            raise ValueError("turn must be a non-negative integer")

        self.messages.append(message)

        explicit_no_preference = {
            _attribute_name(attribute)
            for attribute in (no_preference_attributes or ())
            if _attribute_name(attribute)
        }
        correction_attributes: set[str] = set()
        if parsed_constraints is None:
            if self.interpreter is None:
                parsed = self._adapter.parse_result(
                    message,
                    turn,
                    last_asked_attribute=self.last_asked_attribute,
                )
            else:
                parsed = self.interpreter(message, turn, self.last_asked_attribute)
                correction_attributes.update(parsed.correction_attributes)
            incoming = list(parsed.constraints)
            explicit_no_preference.update(parsed.no_preference_attributes)
        else:
            incoming = list(parsed_constraints)

        self.no_preference_attributes.update(explicit_no_preference)
        self.apply_constraints(
            incoming,
            source_text=message,
            correction=_CORRECTION_RE.search(message) is not None,
            supersede_attributes=correction_attributes,
        )

    def apply_constraints(
        self,
        constraints: Iterable[StructuredConstraint],
        *,
        source_text: str | None = None,
        correction: bool = False,
        supersede_attributes: Iterable[str] | None = None,
    ) -> list[StructuredConstraint]:
        """Apply structured evidence and return records accepted into the state.

        ``correction=True`` supersedes only attributes present in the incoming
        evidence.  ``supersede_attributes`` is an explicit transition hook for
        callers whose interpreter has already resolved correction targets.
        """

        incoming = [self._with_source_text(item, source_text) for item in constraints]
        if not incoming:
            return []

        replacement_attributes = {
            _attribute_name(attribute)
            for attribute in (supersede_attributes or ())
            if _attribute_name(attribute) in ALLOWED_ATTRIBUTES
        }
        if correction:
            replacement_attributes.update(
                constraint.attribute
                for constraint in incoming
                if self._confident_correction_attribute(constraint)
            )
        if replacement_attributes:
            for attribute in replacement_attributes:
                self.supersede_attribute(attribute)

        accepted: list[StructuredConstraint] = []
        for constraint in incoming:
            if correction and not self._confident_correction_attribute(constraint):
                # A V1 catch-all ``feature`` classification can be too weak to
                # identify a correction target.  Preserve the prior state and the
                # raw message instead of clearing unrelated evidence.
                continue
            if constraint.attribute not in ALLOWED_ATTRIBUTES:
                continue
            if constraint.status == "active":
                self.no_preference_attributes.discard(constraint.attribute)
            if self._is_duplicate_active(constraint):
                continue
            self.constraints.append(constraint)
            accepted.append(constraint)
        return accepted

    def _with_source_text(
        self,
        constraint: StructuredConstraint,
        source_text: str | None,
    ) -> StructuredConstraint:
        if not isinstance(constraint, StructuredConstraint):
            raise TypeError("constraints must contain StructuredConstraint values")
        if source_text is not None and not constraint.source_text:
            # ``source_text`` is filled only when an alternate interpreter omitted
            # it; an existing value is always left untouched.
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
        """Reject only clearly vague V1 catch-all correction values.

        V1 maps unmatched text to ``feature``.  A concrete value such as
        ``lightweight`` should still be replaceable, while an override such as
        ``something else`` must not accidentally supersede every existing slot.
        This guard does not add extraction or catalog vocabulary.
        """

        if constraint.attribute != "feature":
            return True
        value = " ".join(constraint.values).strip()
        if value in _VAGUE_CORRECTION_VALUES:
            return False
        if value.startswith("something ") or value.startswith("anything "):
            remainder = value.split(" ", 1)[1].strip()
            return remainder not in _VAGUE_CORRECTION_VALUES
        return True

    def supersede_attribute(self, attribute: str) -> int:
        """Mark all active evidence for one attribute as superseded."""

        normalized_attribute = _attribute_name(attribute)
        if normalized_attribute not in ALLOWED_ATTRIBUTES:
            return 0
        changed = 0
        for constraint in self.constraints:
            if constraint.attribute == normalized_attribute and constraint.active:
                constraint.supersede()
                changed += 1
        return changed

    def active_values(
        self,
        attribute: str | None = None,
        *,
        polarity: str = "include",
    ) -> list[str]:
        """Return unique active values in source order."""

        normalized_attribute = _attribute_name(attribute) if attribute is not None else None
        result: list[str] = []
        for constraint in self.constraints:
            if not constraint.active or constraint.polarity != polarity:
                continue
            if normalized_attribute is not None and constraint.attribute != normalized_attribute:
                continue
            for value in constraint.values:
                if value not in result:
                    result.append(value)
        return result

    def build_query(self) -> str:
        """Compile active positive evidence into the deterministic BM25 query."""

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
            if not constraint.active:
                continue
            if constraint.polarity != "include":
                continue
            if constraint.attribute in self.no_preference_attributes:
                continue
            for value in constraint.values:
                if value not in values:
                    values.append(value)
        return ". ".join(values)

    def choose_question(self) -> str | None:
        """Choose the next attribute using the unchanged V1 fixed order."""

        known = {constraint.attribute for constraint in self.constraints if constraint.active}
        unavailable = known | self.no_preference_attributes | set(self.asked_attributes)
        for attribute in ASK_ORDER:
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
        """Clear conversational and recommendation state for a new session."""

        if session_id is not None:
            self.session_id = session_id
        if user_profile is not None:
            self.user_profile = user_profile
        self.messages.clear()
        self.constraints.clear()
        self.asked_attributes.clear()
        self.no_preference_attributes.clear()
        self.last_asked_attribute = None
        self._adapter = LegacyV1ConstraintAdapter()


__all__ = ["StructuredSessionState"]

from __future__ import annotations

import re

from .state_v2 import StructuredSessionState


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class RawHistorySessionState(StructuredSessionState):
    """Structured session state whose query is the complete raw history."""

    def build_query(self) -> str:
        return ". ".join(self.messages)


class StateConsumedRawHistorySessionState(StructuredSessionState):
    """Raw-history query with state-aware removal and completion of terms."""

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def build_query(self) -> str:
        raw_terms = [
            token
            for message in self.messages
            for token in self._tokens(message)
        ]

        active_positive_terms: list[str] = []
        blocked_terms: set[str] = set()
        for attribute in self.no_preference_attributes:
            blocked_terms.update(self._tokens(attribute))
        for constraint in self.constraints:
            constraint_terms = [
                token
                for value in constraint.values
                for token in self._tokens(value)
            ]
            if (
                constraint.active
                and constraint.polarity == "include"
                and constraint.attribute not in self.no_preference_attributes
            ):
                for term in constraint_terms:
                    if term not in active_positive_terms:
                        active_positive_terms.append(term)
            elif (
                (constraint.active and constraint.polarity == "exclude")
                or (not constraint.active and constraint.polarity == "include")
                or constraint.attribute in self.no_preference_attributes
            ):
                blocked_terms.update(constraint_terms)

        terms = [
            term
            for term in raw_terms
            if term not in blocked_terms or term in active_positive_terms
        ]
        present_terms = set(terms)
        for term in active_positive_terms:
            if term not in present_terms:
                terms.append(term)
                present_terms.add(term)
        return " ".join(terms)


__all__ = ["RawHistorySessionState", "StateConsumedRawHistorySessionState"]

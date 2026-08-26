from __future__ import annotations

from .state_v2 import StructuredSessionState


_LOW_COVERAGE_MAX_ACTIVE_CONSTRAINTS = 3


class RawHistorySessionState(StructuredSessionState):
    """Structured session state whose query is the complete raw history."""

    def build_query(self) -> str:
        return ". ".join(self.messages)


class CoverageAdaptiveSessionState(StructuredSessionState):
    """Use raw history for low-coverage corrections and state otherwise."""

    def build_query(self) -> str:
        state_query = super().build_query()
        raw_history = ". ".join(self.messages)
        has_superseded = any(not constraint.active for constraint in self.constraints)
        if (
            has_superseded
            and len(self.active_constraints) <= _LOW_COVERAGE_MAX_ACTIVE_CONSTRAINTS
        ):
            return raw_history
        return state_query or raw_history


__all__ = ["RawHistorySessionState", "CoverageAdaptiveSessionState"]

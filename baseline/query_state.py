from __future__ import annotations

from .state_v2 import StructuredSessionState


class RawHistorySessionState(StructuredSessionState):
    """Structured session state whose query is the complete raw history."""

    def build_query(self) -> str:
        return ". ".join(self.messages)


class StatePrioritizedRawHistorySessionState(StructuredSessionState):
    """Prioritize active state evidence while retaining the exact raw history."""

    def build_query(self) -> str:
        state_query = super().build_query()
        raw_history = ". ".join(self.messages)
        if state_query and raw_history:
            return f"{state_query}. {raw_history}"
        return state_query or raw_history


__all__ = ["RawHistorySessionState", "StatePrioritizedRawHistorySessionState"]

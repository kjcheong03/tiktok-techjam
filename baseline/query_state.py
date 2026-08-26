from __future__ import annotations

from .state_v2 import StructuredSessionState


class RawHistorySessionState(StructuredSessionState):
    """Structured session state whose query is the complete raw history."""

    def build_query(self) -> str:
        return ". ".join(self.messages)


__all__ = ["RawHistorySessionState"]

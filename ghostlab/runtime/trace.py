from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RuntimeTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: str
    turn: int = Field(ge=1, le=10)
    policy_id: str
    ask_attribute: str | None
    retrieval_route: str
    query_sha256: str
    candidate_count: int = Field(ge=0)
    top_ids: tuple[str, ...]
    fallback_reason: str | None = None

    @classmethod
    def from_observable(
        cls,
        *,
        session_id: str,
        turn: int,
        policy_id: str,
        ask_attribute: str | None,
        retrieval_route: str,
        query: str,
        top_ids: list[str],
        fallback_reason: str | None = None,
    ) -> RuntimeTrace:
        return cls(
            session_id=session_id,
            turn=turn,
            policy_id=policy_id,
            ask_attribute=ask_attribute,
            retrieval_route=retrieval_route,
            query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            candidate_count=len(top_ids),
            top_ids=tuple(top_ids),
            fallback_reason=fallback_reason,
        )


class JsonlTraceSink:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trace: RuntimeTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(trace.model_dump_json() + "\n")

    def read(self) -> list[RuntimeTrace]:
        if not self.path.exists():
            return []
        return [
            RuntimeTrace.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def assert_trace_has_no_research_labels(trace: RuntimeTrace) -> None:
    serialized = json.dumps(trace.model_dump()).casefold()
    forbidden = ("ground_truth", "intent_card", "scenario_type", "target_id", "reward")
    if any(name in serialized for name in forbidden):
        raise ValueError("research-only label detected in runtime trace")

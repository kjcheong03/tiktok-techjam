from __future__ import annotations

from pathlib import Path

from ghostlab.runtime.selected import SelectedRuntime


class Agent:
    """Official TechJam adapter; implementation lives in the submission runtime."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._runtime = SelectedRuntime(catalog_path)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._runtime.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        return self._runtime.respond(session_id, user_message, turn, top_k)

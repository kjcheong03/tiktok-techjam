from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath

from ghostlab.competition.contract import AgentProtocol
from ghostlab.research.technique_suite import build_suite_agent, load_suite_config
from ghostlab.runtime.adaptive_factory import build_adaptive_hybrid_agent
from ghostlab.runtime.agent import GhostLabRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_POINTER = PROJECT_ROOT / "configs/active_candidate.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_active_preset(pointer_path: Path = ACTIVE_POINTER) -> Path | None:
    if not pointer_path.is_file():
        return None
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported active-candidate pointer schema")
    relative = payload.get("preset_path")
    if not isinstance(relative, str):
        raise TypeError("active-candidate pointer requires preset_path")
    safe = PurePath(relative)
    if safe.is_absolute() or ".." in safe.parts or not safe.name:
        raise ValueError("active preset must stay inside the project")
    preset = (PROJECT_ROOT / safe).resolve()
    preset.relative_to(PROJECT_ROOT.resolve())
    if not preset.is_file():
        raise FileNotFoundError(f"active preset is missing: {relative}")
    actual = sha256_file(preset)
    if actual != payload.get("preset_sha256"):
        raise ValueError("active preset hash does not match its pointer")
    return preset


class SelectedRuntime:
    """Use an explicitly activated suite, otherwise preserve the frozen champion."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        pointer_path: Path = ACTIVE_POINTER,
    ) -> None:
        self._runtime: AgentProtocol
        self._fallback: GhostLabRuntime | None
        preset = resolve_active_preset(pointer_path)
        if preset is None:
            self._runtime = GhostLabRuntime(catalog_path)
            self._fallback = None
        else:
            payload = json.loads(preset.read_text(encoding="utf-8"))
            if payload.get("architecture") == "adaptive_hybrid_1a_3b_v1":
                self._runtime = build_adaptive_hybrid_agent(
                    catalog_path,
                    config_path=preset,
                    project_root=PROJECT_ROOT,
                )
            else:
                self._runtime = build_suite_agent(
                    load_suite_config(preset), catalog_path
                )
            self._fallback = GhostLabRuntime(catalog_path)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._runtime.reset(session_id, user_profile)
        if self._fallback is not None:
            self._fallback.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        try:
            return self._runtime.respond(session_id, user_message, turn, top_k)
        except Exception:
            if self._fallback is None:
                raise
            return self._fallback.respond(session_id, user_message, turn, top_k)

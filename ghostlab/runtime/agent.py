from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from baseline.agent import BaselineAgent
from baseline.retrieval import DenseRetriever, KeywordRetriever
from ghostlab.competition.contract import AgentProtocol
from ghostlab.policy.models import RuntimeConfig
from ghostlab.runtime.compiled import CompiledKeywordAgent
from ghostlab.runtime.normalizer import normalize_response

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/compiled_policy.json"


def _catalog_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {str(json.loads(line)["parent_asin"]) for line in handle if line.strip()}


class GhostLabRuntime:
    """Submission runtime with lazy optional dependencies and deterministic fallback."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config_path: str | Path = DEFAULT_CONFIG,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = RuntimeConfig.model_validate_json(
            Path(config_path).read_text(encoding="utf-8")
        )
        self.catalog_ids = _catalog_ids(self.catalog_path)
        techniques = self.config.techniques
        self._primary: AgentProtocol
        if techniques.state_mode in {"multi", "compressed", "raw_history"}:
            self._primary = CompiledKeywordAgent(
                self.catalog_path, self.catalog_ids, techniques
            )
        else:
            keyword = KeywordRetriever(self.catalog_path)
            dense = None
            if techniques.retrieval_route in {"dense", "rrf", "weighted_fusion"}:
                dense = DenseRetriever(
                    self.catalog_path, model_name=techniques.dense_model
                )
            mode: Literal["keyword", "dense", "hybrid"]
            if techniques.retrieval_route in {"rrf", "weighted_fusion"}:
                mode = "hybrid"
            elif techniques.retrieval_route == "dense":
                mode = "dense"
            else:
                mode = "keyword"
            self._primary = BaselineAgent(
                mode=mode,
                stateful=techniques.state_mode != "off",
                keyword=keyword,
                dense=dense,
                retrieval_k=techniques.retrieval_k,
                rrf_constant=techniques.rrf_constant,
            )
        self._fallback = BaselineAgent(
            mode="keyword",
            stateful=False,
            keyword=KeywordRetriever(self.catalog_path),
            dense=None,
            retrieval_k=techniques.retrieval_k,
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._primary.reset(session_id, user_profile)
        self._fallback.reset(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be non-empty")
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string")
        if not 1 <= turn <= 10 or top_k != 10:
            raise ValueError("turn/top_k outside official contract")
        try:
            payload = self._primary.respond(session_id, user_message, turn, top_k)
            return normalize_response(payload, self.catalog_ids, top_k)
        except Exception:  # noqa: BLE001 - contract boundary must always degrade safely
            payload = self._fallback.respond(session_id, user_message, turn, top_k)
            return normalize_response(payload, self.catalog_ids, top_k)

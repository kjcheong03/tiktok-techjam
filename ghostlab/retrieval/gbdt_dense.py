from __future__ import annotations

import time
from pathlib import Path

from ghostlab.retrieval.dense import DenseIndex
from ghostlab.retrieval.fusion import sparse_first_union_ids
from ghostlab.retrieval.gbdt import LambdaMARTReranker
from ghostlab.retrieval.quality import CatalogQualityReranker
from ghostlab.retrieval.query import DenseQueryVariant, build_dense_query
from ghostlab.retrieval.sparse import SparseIndex
from ghostlab.runtime.normalizer import normalize_response
from ghostlab.state.memory import ConversationState


class DeepGBDTAgent:
    """Frozen sparse or sparse-first dense pool with fold-local deep GBDT."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        sparse: SparseIndex,
        dense: DenseIndex | None,
        quality: CatalogQualityReranker,
        reranker: LambdaMARTReranker,
        field_weights: tuple[float, float, float, float, float, float],
        question_order: tuple[str, ...],
        dense_query_variant: DenseQueryVariant | None,
    ) -> None:
        self.sparse = sparse
        self.dense = dense
        self.quality = quality
        self.reranker = reranker
        self.field_weights = field_weights
        self.question_order = question_order
        self.dense_query_variant = dense_query_variant
        self.catalog_ids = self._read_ids(catalog_path)
        self.sessions: dict[str, ConversationState] = {}
        self.latencies_ms: list[float] = []
        self.failure_count = 0

    @staticmethod
    def _read_ids(path: str | Path) -> set[str]:
        import json

        with Path(path).open(encoding="utf-8") as handle:
            return {
                str(json.loads(line)["parent_asin"]) for line in handle if line.strip()
            }

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = ConversationState(session_id, user_profile)

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> dict:
        started = time.perf_counter()
        try:
            state = self.sessions[session_id]
            state.observe(user_message, turn)
            raw_query = ". ".join(state.messages)
            sparse_ids = [
                item.parent_asin
                for item in self.sparse.search(raw_query, 200, self.field_weights).items
            ]
            if self.dense is None:
                if self.dense_query_variant is not None:
                    raise RuntimeError("sparse-only arm cannot specify a dense query")
                candidates = sparse_ids
            else:
                if self.dense_query_variant is None:
                    raise RuntimeError("dense arm requires a query variant")
                dense_query = build_dense_query(state, self.dense_query_variant)
                dense_ids = [
                    item.parent_asin
                    for item in self.dense.search(dense_query, 200).items
                ]
                candidates = sparse_first_union_ids(sparse_ids, dense_ids, limit=400)
                if candidates[: len(sparse_ids)] != sparse_ids:
                    raise RuntimeError("sparse-first union changed the sparse head")
            depth = len(candidates)
            ranked = self.quality.rerank(candidates, weight=0.2, rerank_k=depth)
            ranked = self.reranker.rerank(raw_query, ranked, rerank_k=depth)
            question = (
                self.question_order[turn - 1]
                if turn <= len(self.question_order)
                else None
            )
            state.last_asked_attribute = question
            if question is not None:
                state.asked_attributes.append(question)
            return normalize_response(
                {
                    "message": (
                        "Here are the closest matches based on what you have shared."
                        if question is None
                        else "Do you have a preference for "
                        f"{question.replace('_', ' ')}?"
                    ),
                    "ask_attribute": question,
                    "recommendations": ranked,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
                self.catalog_ids,
                top_k,
            )
        except Exception:
            self.failure_count += 1
            raise
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1000.0)

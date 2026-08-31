from __future__ import annotations

from pathlib import Path

import numpy as np

from ghostlab.retrieval.late_interaction import (
    LateInteractionFeasibility,
    LateInteractionReranker,
    LateInteractionRetriever,
    TokenEmbeddingStore,
    maxsim_score,
)


class StaticTokenEncoder:
    def encode_tokens(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32) for _ in texts]


def test_maxsim_and_bounded_reranking() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    exact = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    partial = np.asarray([[1.0, 0.0]], dtype=np.float32)
    assert maxsim_score(query, exact) == 2.0
    assert maxsim_score(query, partial) == 1.0
    reranker = LateInteractionReranker(
        StaticTokenEncoder(), {"partial": partial, "exact": exact}
    )
    assert reranker.rerank("query", ["partial", "exact", "tail"], rerank_k=2) == [
        "exact",
        "partial",
        "tail",
    ]


def test_token_store_round_trip_and_reference_retrieval(tmp_path: Path) -> None:
    store = TokenEmbeddingStore.from_documents(
        ["partial", "exact"],
        [
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ],
    )
    path = tmp_path / "tokens.npz"
    store.save(path)
    loaded = TokenEmbeddingStore.load(path)
    retriever = LateInteractionRetriever(StaticTokenEncoder(), loaded)
    assert retriever.search("query", 2) == ["exact", "partial"]


def test_flat_50k_feasibility_fails_conservative_compute_gate() -> None:
    estimate = LateInteractionFeasibility(
        document_count=50_000,
        average_document_tokens=64,
        embedding_dimension=128,
        dtype_bytes=2,
        query_tokens=24,
    )
    assert estimate.index_gib < 1.0
    assert estimate.dot_products_per_query == 76_800_000
    assert not estimate.passes(max_index_gib=1.0, max_dot_products_per_query=50_000_000)

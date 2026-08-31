from __future__ import annotations

import json
from pathlib import Path

from ghostlab.retrieval.sparse import query_terms


class LinearLexicalReranker:
    """Compact rank-plus-token-overlap reranker with deterministic ties."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.terms: dict[str, set[str]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                text = " ".join(
                    str(product.get(field) or "")
                    for field in ("title", "categories", "features")
                )
                self.terms[str(product["parent_asin"])] = set(query_terms(text, 200))

    def rerank(self, query: str, ranking: list[str], rerank_k: int = 50) -> list[str]:
        query_set = set(query_terms(query, 80))
        head = ranking[:rerank_k]
        count = len(head)
        scores = {}
        for rank, identifier in enumerate(head, start=1):
            base = 1.0 if count == 1 else 1.0 - (rank - 1) / max(1, count - 1)
            terms = self.terms.get(identifier, set())
            overlap = len(query_set & terms) / max(1, len(query_set | terms))
            scores[identifier] = 0.8 * base + 0.2 * overlap
        ordered = sorted(head, key=lambda item: (-scores[item], item))
        return [*ordered, *ranking[rerank_k:]]

from __future__ import annotations

import json
from pathlib import Path

from ghostlab.retrieval.sparse import query_terms


class ProfilePriorReranker:
    """Small, deterministic soft prior from anonymized preference tags."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.product_terms: dict[str, frozenset[str]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                text = " ".join(
                    str(product.get(field) or "")
                    for field in (
                        "title",
                        "categories",
                        "features",
                        "details",
                        "description",
                    )
                )
                self.product_terms[str(product["parent_asin"])] = frozenset(
                    query_terms(text, 300)
                )
        self.profile_terms: dict[str, frozenset[str]] = {}

    def reset(self, session_id: str, user_profile: dict[str, object]) -> None:
        raw_tags = user_profile.get("preference_tags")
        tags = raw_tags if isinstance(raw_tags, (list, tuple, set)) else []
        text = " ".join(str(value) for value in tags)
        self.profile_terms[session_id] = frozenset(query_terms(text, 40))

    def rerank(
        self,
        session_id: str,
        ranking: list[str],
        *,
        weight: float,
        rerank_k: int = 50,
    ) -> list[str]:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("profile prior weight must be between zero and one")
        terms = self.profile_terms.get(session_id, frozenset())
        if not terms or weight == 0.0:
            return list(ranking)
        head = ranking[:rerank_k]
        count = len(head)
        scores = {}
        original_ranks = {}
        for rank, identifier in enumerate(head, start=1):
            base = 1.0 if count == 1 else 1.0 - (rank - 1) / max(1, count - 1)
            product_terms = self.product_terms.get(identifier, frozenset())
            overlap = len(terms & product_terms) / len(terms)
            scores[identifier] = base + weight * overlap
            original_ranks[identifier] = rank
        ordered = sorted(
            head,
            key=lambda identifier: (
                -scores[identifier],
                original_ranks[identifier],
                identifier,
            ),
        )
        return [*ordered, *ranking[rerank_k:]]

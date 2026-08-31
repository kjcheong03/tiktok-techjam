from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ghostlab.retrieval.sparse import query_terms


@dataclass(frozen=True)
class CategoryHit:
    parent_asin: str
    score: float
    rank: int


class CategoryCandidateIndex:
    """Independent category-only candidate source for the 1B merge."""

    def __init__(self, catalog_path: str | Path) -> None:
        postings: dict[str, list[str]] = defaultdict(list)
        product_terms: dict[str, frozenset[str]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = str(row["parent_asin"])
                categories = " ".join(str(item) for item in row.get("categories") or ())
                terms = frozenset(query_terms(categories, 80))
                product_terms[identifier] = terms
                for term in terms:
                    postings[term].append(identifier)
        self.product_terms = product_terms
        self.postings = {term: tuple(values) for term, values in postings.items()}
        self.document_count = len(product_terms)

    def search(
        self,
        query: str,
        *,
        limit: int,
        preferred_categories: Sequence[str] = (),
    ) -> tuple[CategoryHit, ...]:
        if limit <= 0:
            raise ValueError("category search limit must be positive")
        category_text = " ".join(preferred_categories).strip()
        terms = query_terms(category_text or query, 40)
        scores: dict[str, float] = defaultdict(float)
        for term in terms:
            posting = self.postings.get(term, ())
            if not posting:
                continue
            inverse_frequency = math.log1p(self.document_count / len(posting))
            for identifier in posting:
                scores[identifier] += inverse_frequency
        ordered = sorted(scores, key=lambda item: (-scores[item], item))[:limit]
        maximum = scores[ordered[0]] if ordered else 1.0
        return tuple(
            CategoryHit(
                parent_asin=identifier,
                score=scores[identifier] / maximum,
                rank=rank,
            )
            for rank, identifier in enumerate(ordered, start=1)
        )


__all__ = ["CategoryCandidateIndex", "CategoryHit"]

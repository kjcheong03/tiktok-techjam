from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ghostlab.retrieval.sparse import query_terms
from ghostlab.state.query_expansion import ExpansionGuard, ExpansionTerm, QueryExpansion


def _catalog_text(product: dict[str, object]) -> str:
    return " ".join(
        str(product.get(field) or "")
        for field in ("title", "categories", "features", "details", "store")
    )


class CatalogPseudoRelevanceFeedback:
    """Deterministic IDF-weighted PRF grounded only in retrieved catalog products."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        feedback_k: int = 5,
        minimum_support: float = 0.4,
        max_terms: int = 4,
        max_added_ratio: float = 0.5,
    ) -> None:
        if feedback_k <= 0:
            raise ValueError("feedback_k must be positive")
        if not 0.0 < minimum_support <= 1.0:
            raise ValueError("minimum_support must be in (0, 1]")
        self.feedback_k = feedback_k
        self.minimum_support = minimum_support
        self.guard = ExpansionGuard(
            max_terms=max_terms, max_added_ratio=max_added_ratio
        )
        self.terms: dict[str, tuple[str, ...]] = {}
        document_frequency: Counter[str] = Counter()
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                identifier = str(product["parent_asin"])
                terms = tuple(query_terms(_catalog_text(product), limit=300))
                self.terms[identifier] = terms
                document_frequency.update(set(terms))
        self.document_count = len(self.terms)
        self.idf = {
            term: math.log((self.document_count + 1) / (count + 1)) + 1.0
            for term, count in document_frequency.items()
        }

    def expand(self, query: str, ranking: list[str]) -> QueryExpansion:
        feedback = [
            identifier
            for identifier in ranking[: self.feedback_k]
            if identifier in self.terms
        ]
        if not query.strip() or len(feedback) < 2:
            return QueryExpansion(query, (), "insufficient_feedback")
        query_set = set(query_terms(query, 100))
        support: Counter[str] = Counter()
        weighted_rank: dict[str, float] = defaultdict(float)
        for rank, identifier in enumerate(feedback, start=1):
            for term in set(self.terms[identifier]):
                if term in query_set:
                    continue
                support[term] += 1
                weighted_rank[term] += 1.0 / rank
        proposed: list[ExpansionTerm] = []
        for term, count in support.items():
            support_ratio = count / len(feedback)
            if support_ratio < self.minimum_support:
                continue
            rank_agreement = weighted_rank[term] / sum(
                1.0 / rank for rank in range(1, len(feedback) + 1)
            )
            idf = self.idf.get(term, 0.0)
            idf_confidence = min(1.0, idf / 8.0)
            confidence = support_ratio * rank_agreement * idf_confidence
            proposed.append(
                ExpansionTerm(term, max(0.0, min(1.0, confidence)), "catalog_prf")
            )
        accepted = self.guard.apply(query, proposed)
        reason = "expanded" if accepted else "low_agreement_or_guarded"
        return QueryExpansion(query, accepted, reason)

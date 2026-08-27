from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ghostlab.retrieval.sparse import query_terms


@dataclass(frozen=True)
class DiversificationContext:
    turn: int
    active_constraint_count: int


@dataclass(frozen=True)
class DiversificationDecision:
    ranking: list[str]
    activated: bool
    reason: str


class FacetMMRDiversifier:
    """Conditional, bounded MMR over catalog metadata; never changes candidate set."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        relevance_weight: float = 0.85,
        rerank_k: int = 30,
        output_k: int = 10,
        max_turn: int = 2,
        max_active_constraints: int = 1,
        preserve_first: bool = True,
    ) -> None:
        if not 0.0 <= relevance_weight <= 1.0:
            raise ValueError("relevance_weight must be between zero and one")
        if rerank_k <= 0 or output_k <= 0 or output_k > rerank_k:
            raise ValueError("invalid diversification depths")
        if max_turn <= 0 or max_active_constraints < 0:
            raise ValueError("invalid diversification activation bounds")
        self.relevance_weight = relevance_weight
        self.rerank_k = rerank_k
        self.output_k = output_k
        self.max_turn = max_turn
        self.max_active_constraints = max_active_constraints
        self.preserve_first = preserve_first
        self.facets: dict[str, frozenset[str]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                text = " ".join(
                    str(product.get(field) or "")
                    for field in ("categories", "details", "features", "store")
                )
                self.facets[str(product["parent_asin"])] = frozenset(
                    query_terms(text, 200)
                )

    def similarity(self, left: str, right: str) -> float:
        left_facets = self.facets.get(left, frozenset())
        right_facets = self.facets.get(right, frozenset())
        union = left_facets | right_facets
        return len(left_facets & right_facets) / len(union) if union else 0.0

    def rerank(
        self, ranking: list[str], context: DiversificationContext
    ) -> DiversificationDecision:
        if context.turn > self.max_turn:
            return DiversificationDecision(list(ranking), False, "late_turn")
        if context.active_constraint_count > self.max_active_constraints:
            return DiversificationDecision(list(ranking), False, "specific_intent")
        head = list(dict.fromkeys(ranking[: self.rerank_k]))
        if len(head) < 2:
            return DiversificationDecision(
                list(ranking), False, "insufficient_candidates"
            )
        original_rank = {identifier: rank for rank, identifier in enumerate(head)}
        rank_denominator = max(1, len(head) - 1)
        selected = [head[0]] if self.preserve_first else []
        remaining = head[1:] if self.preserve_first else head[:]
        while remaining and len(selected) < min(self.output_k, len(head)):

            def score(identifier: str) -> tuple[float, int, str]:
                relevance = 1.0 - original_rank[identifier] / rank_denominator
                redundancy = max(
                    (self.similarity(identifier, prior) for prior in selected),
                    default=0.0,
                )
                mmr = (
                    self.relevance_weight * relevance
                    - (1.0 - self.relevance_weight) * redundancy
                )
                return mmr, -original_rank[identifier], identifier

            chosen = max(remaining, key=score)
            selected.append(chosen)
            remaining.remove(chosen)
        reordered_head = [*selected, *remaining]
        return DiversificationDecision(
            [*reordered_head, *ranking[self.rerank_k :]], True, "early_uncertain"
        )

    def facet_coverage(self, ranking: list[str], k: int = 10) -> int:
        return len(
            set().union(
                *(
                    self.facets.get(identifier, frozenset())
                    for identifier in ranking[:k]
                )
            )
        )

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ghostlab.retrieval.category import CategoryHit

IntentRoute = Literal["buying", "browsing"]
EvidenceSource = Literal["keyword", "category", "vector"]


@dataclass(frozen=True)
class CandidateEvidence:
    parent_asin: str
    sources: frozenset[EvidenceSource]
    keyword_rank: int | None
    keyword_score: float | None
    category_rank: int | None
    category_score: float | None
    vector_rank: int | None
    vector_score: float | None
    aggregate_score: float


@dataclass(frozen=True)
class MergedCandidatePool:
    route: IntentRoute
    candidates: tuple[CandidateEvidence, ...]

    @property
    def identifiers(self) -> list[str]:
        return [item.parent_asin for item in self.candidates]

    def contribution_counts(self) -> dict[str, int]:
        return {
            source: sum(source in item.sources for item in self.candidates)
            for source in ("keyword", "category", "vector")
        }

    def retain(self, identifiers: Sequence[str]) -> MergedCandidatePool:
        allowed = set(identifiers)
        by_id = {item.parent_asin: item for item in self.candidates}
        return MergedCandidatePool(
            route=self.route,
            candidates=tuple(
                by_id[identifier]
                for identifier in identifiers
                if identifier in allowed and identifier in by_id
            ),
        )


def _percentile(rank: int, count: int) -> float:
    return 1.0 if count <= 1 else 1.0 - (rank - 1) / (count - 1)


def merge_candidate_routes(
    *,
    route: IntentRoute,
    keyword_ids: Sequence[str],
    category_hits: Sequence[CategoryHit],
    vector_ids: Sequence[str],
    limit: int,
    keyword_weight: float,
    category_weight: float,
    vector_weight: float,
    strategy: Literal["weighted", "rrf", "sparse_first_union"] = "weighted",
    rrf_constant: int = 60,
    keyword_scores: Mapping[str, float] | None = None,
    vector_scores: Mapping[str, float] | None = None,
) -> MergedCandidatePool:
    """Merge three-source evidence while preserving exact source provenance."""
    if limit <= 0:
        raise ValueError("merged candidate limit must be positive")
    if any(weight < 0.0 for weight in (keyword_weight, category_weight, vector_weight)):
        raise ValueError("merge weights must be non-negative")
    if strategy not in {"weighted", "rrf", "sparse_first_union"}:
        raise ValueError(f"unknown merge strategy: {strategy}")
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be positive")

    keyword = list(dict.fromkeys(keyword_ids))
    vector = list(dict.fromkeys(vector_ids))
    category = list(dict.fromkeys(item.parent_asin for item in category_hits))
    keyword_ranks = {item: rank for rank, item in enumerate(keyword, start=1)}
    vector_ranks = {item: rank for rank, item in enumerate(vector, start=1)}
    category_ranks = {item: rank for rank, item in enumerate(category, start=1)}
    category_scores = {item.parent_asin: item.score for item in category_hits}

    identifiers = list(dict.fromkeys([*keyword, *category, *vector]))
    values: list[CandidateEvidence] = []
    for identifier in identifiers:
        score = 0.0
        sources: set[EvidenceSource] = set()
        if identifier in keyword_ranks:
            sources.add("keyword")
            keyword_relevance = (
                keyword_scores.get(identifier, 0.0)
                if keyword_scores is not None
                else _percentile(keyword_ranks[identifier], len(keyword))
            )
            score += (
                keyword_weight / (rrf_constant + keyword_ranks[identifier])
                if strategy == "rrf"
                else keyword_weight * keyword_relevance
            )
        if identifier in category_ranks:
            sources.add("category")
            score += (
                category_weight / (rrf_constant + category_ranks[identifier])
                if strategy == "rrf"
                else category_weight * category_scores[identifier]
            )
        if identifier in vector_ranks:
            sources.add("vector")
            relevance = (
                vector_scores.get(identifier, 0.0)
                if vector_scores is not None
                else _percentile(vector_ranks[identifier], len(vector))
            )
            score += (
                vector_weight / (rrf_constant + vector_ranks[identifier])
                if strategy == "rrf"
                else vector_weight * relevance
            )
        if strategy == "sparse_first_union":
            # Preserve the precision route as the primary ordering signal while
            # still admitting independently discovered category/vector products.
            if identifier in keyword_ranks:
                score += 2.0 + _percentile(
                    keyword_ranks[identifier], len(keyword)
                )
            elif identifier in category_ranks:
                score += 1.0 + 0.5 * _percentile(
                    category_ranks[identifier], len(category)
                )
        values.append(
            CandidateEvidence(
                parent_asin=identifier,
                sources=frozenset(sources),
                keyword_rank=keyword_ranks.get(identifier),
                keyword_score=(
                    keyword_scores.get(identifier)
                    if keyword_scores is not None
                    else (
                        _percentile(keyword_ranks[identifier], len(keyword))
                        if identifier in keyword_ranks
                        else None
                    )
                ),
                category_rank=category_ranks.get(identifier),
                category_score=category_scores.get(identifier),
                vector_rank=vector_ranks.get(identifier),
                vector_score=(
                    vector_scores.get(identifier)
                    if vector_scores is not None
                    else (
                        _percentile(vector_ranks[identifier], len(vector))
                        if identifier in vector_ranks
                        else None
                    )
                ),
                aggregate_score=score,
            )
        )
    positions = {item: rank for rank, item in enumerate(identifiers)}
    ordered = sorted(
        values,
        key=lambda item: (
            -item.aggregate_score,
            positions[item.parent_asin],
            item.parent_asin,
        ),
    )[:limit]
    return MergedCandidatePool(route=route, candidates=tuple(ordered))


__all__ = [
    "CandidateEvidence",
    "EvidenceSource",
    "IntentRoute",
    "MergedCandidatePool",
    "merge_candidate_routes",
]

from __future__ import annotations

from collections.abc import Sequence

from ghostlab.policy.models import RankedCandidate, RankedCandidates


def fuse_rankings(
    sparse: RankedCandidates,
    dense: RankedCandidates,
    *,
    sparse_weight: float = 0.75,
    dense_weight: float = 0.25,
    limit: int = 200,
) -> RankedCandidates:
    if abs(sparse_weight + dense_weight - 1.0) > 1e-9:
        raise ValueError("fusion weights must sum to one")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in ((sparse, sparse_weight), (dense, dense_weight)):
        for item in ranking.items:
            score = item.normalized_score if item.normalized_score is not None else 0.0
            scores[item.parent_asin] = (
                scores.get(item.parent_asin, 0.0) + weight * score
            )
            best_rank[item.parent_asin] = min(
                best_rank.get(item.parent_asin, item.rank), item.rank
            )
    ordered = sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))[
        :limit
    ]
    items = tuple(
        RankedCandidate(
            parent_asin=identifier,
            route="weighted_fusion",
            rank=rank,
            raw_score=scores[identifier],
            normalized_score=scores[identifier],
        )
        for rank, identifier in enumerate(ordered, start=1)
    )
    return RankedCandidates(
        items=items,
        route="weighted_fusion",
        requested_k=limit,
        elapsed_ms=sparse.elapsed_ms + dense.elapsed_ms,
    )


def jaccard_at(left: Sequence[str], right: Sequence[str], k: int) -> float | None:
    left_set, right_set = set(left[:k]), set(right[:k])
    if not left_set or not right_set:
        return None
    return len(left_set & right_set) / len(left_set | right_set)


def weighted_fuse_ids(
    sparse: Sequence[str],
    dense: Sequence[str],
    *,
    sparse_weight: float = 0.75,
    dense_weight: float = 0.25,
    limit: int = 200,
) -> list[str]:
    if abs(sparse_weight + dense_weight - 1.0) > 1e-9:
        raise ValueError("fusion weights must sum to one")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in ((sparse, sparse_weight), (dense, dense_weight)):
        count = len(ranking)
        for rank, identifier in enumerate(ranking, start=1):
            percentile = 1.0 if count == 1 else 1.0 - (rank - 1) / max(1, count - 1)
            scores[identifier] = scores.get(identifier, 0.0) + weight * percentile
            best_rank[identifier] = min(best_rank.get(identifier, rank), rank)
    return sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))[
        :limit
    ]

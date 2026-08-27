from __future__ import annotations

from collections.abc import Sequence


def sparse_semantic_union_ids(
    sparse: Sequence[str],
    semantic: Sequence[str],
    *,
    sparse_weight: float = 0.75,
    semantic_weight: float = 0.25,
    rank_constant: int = 60,
    limit: int = 200,
) -> list[str]:
    """Weighted RRF union with deterministic ties and no duplicate candidates."""

    if limit <= 0 or rank_constant <= 0:
        raise ValueError("fusion bounds must be positive")
    if sparse_weight < 0.0 or semantic_weight < 0.0:
        raise ValueError("fusion weights cannot be negative")
    if abs(sparse_weight + semantic_weight - 1.0) > 1e-9:
        raise ValueError("fusion weights must sum to one")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in ((sparse, sparse_weight), (semantic, semantic_weight)):
        for rank, identifier in enumerate(dict.fromkeys(ranking), start=1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (
                rank_constant + rank
            )
            best_rank[identifier] = min(best_rank.get(identifier, rank), rank)
    return sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))[
        :limit
    ]

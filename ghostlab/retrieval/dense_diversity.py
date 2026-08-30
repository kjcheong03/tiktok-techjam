from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def max_relevance_select(
    candidate_ids: Sequence[str],
    relevance_scores: Mapping[str, float],
    *,
    output_k: int,
) -> list[str]:
    """Select unique candidates by their maximum relevance across query views."""
    if output_k < 0:
        raise ValueError("output_k must be non-negative")
    unique_ids = list(dict.fromkeys(candidate_ids))
    missing = [item for item in unique_ids if item not in relevance_scores]
    if missing:
        raise ValueError(f"missing relevance score for {missing[0]}")
    if not all(np.isfinite(float(relevance_scores[item])) for item in unique_ids):
        raise ValueError("relevance scores must be finite")
    positions = {item: index for index, item in enumerate(unique_ids)}
    return sorted(
        unique_ids,
        key=lambda item: (-float(relevance_scores[item]), positions[item], item),
    )[:output_k]


def embedding_mmr_select(
    candidate_ids: Sequence[str],
    relevance_scores: Sequence[float] | Mapping[str, float],
    product_embeddings: Mapping[str, ArrayLike],
    *,
    output_k: int,
    relevance_weight: float = 0.85,
) -> list[str]:
    """Select a deterministic unique subset using embedding cosine MMR."""
    if output_k < 0:
        raise ValueError("output_k must be non-negative")
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be between zero and one")
    if not isinstance(relevance_scores, Mapping) and len(relevance_scores) != len(
        candidate_ids
    ):
        raise ValueError("candidate_ids and relevance_scores must have equal length")

    unique_ids: list[str] = []
    unique_scores: list[float] = []
    seen: set[str] = set()
    for index, identifier in enumerate(candidate_ids):
        if identifier in seen:
            continue
        seen.add(identifier)
        unique_ids.append(identifier)
        score = (
            relevance_scores[identifier]
            if isinstance(relevance_scores, Mapping)
            else relevance_scores[index]
        )
        unique_scores.append(float(score))

    count = min(output_k, len(unique_ids))
    if count == 0:
        return []
    missing = [item for item in unique_ids if item not in product_embeddings]
    if missing:
        raise ValueError(f"missing product embedding for {missing[0]}")
    embeddings = np.stack(
        [np.asarray(product_embeddings[item], dtype=np.float32) for item in unique_ids]
    )
    scores = np.asarray(unique_scores, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError("product embeddings must be one-dimensional vectors")
    if not np.all(np.isfinite(embeddings)) or not np.all(np.isfinite(scores)):
        raise ValueError("scores and product embeddings must be finite")

    selected: list[int] = []
    remaining = set(range(len(unique_ids)))
    maximum_similarity: np.ndarray | None = None
    while remaining and len(selected) < count:
        if maximum_similarity is None:
            mmr_scores = relevance_weight * scores
        else:
            mmr_scores = (
                relevance_weight * scores
                - (1.0 - relevance_weight) * maximum_similarity
            )
        chosen = min(
            remaining,
            key=lambda index: (-float(mmr_scores[index]), index, unique_ids[index]),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        similarities = embeddings @ embeddings[chosen]
        maximum_similarity = (
            similarities
            if maximum_similarity is None
            else np.maximum(maximum_similarity, similarities)
        )
    return [unique_ids[index] for index in selected]


__all__ = ["embedding_mmr_select", "max_relevance_select"]

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalSignals:
    candidate_count: int
    top1_margin: float | None
    normalized_entropy: float | None
    sparse_dense_top10_overlap: float | None


def retrieval_signals(
    scores: Sequence[float],
    *,
    sparse_ids: Sequence[str] | None = None,
    dense_ids: Sequence[str] | None = None,
) -> RetrievalSignals:
    finite = [float(score) for score in scores if math.isfinite(float(score))]
    margin = finite[0] - finite[1] if len(finite) >= 2 else None
    if len(finite) < 2:
        entropy = None
    else:
        minimum = min(finite)
        weights = [score - minimum + 1e-12 for score in finite]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        entropy = -sum(value * math.log(value) for value in probabilities) / math.log(
            len(probabilities)
        )
    if sparse_ids is None or dense_ids is None:
        overlap = None
    else:
        sparse_set, dense_set = set(sparse_ids[:10]), set(dense_ids[:10])
        union = sparse_set | dense_set
        overlap = len(sparse_set & dense_set) / len(union) if union else 0.0
    return RetrievalSignals(len(finite), margin, entropy, overlap)

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


class CatalogQualityReranker:
    """Bayesian rating and popularity as a bounded soft ranking prior."""

    def __init__(
        self, catalog_path: str | Path, *, prior_strength: float = 50.0
    ) -> None:
        if prior_strength <= 0.0:
            raise ValueError("prior strength must be positive")
        rows = []
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                rating = product.get("average_rating")
                count = product.get("rating_number")
                rows.append(
                    (
                        str(product["parent_asin"]),
                        float(rating) if isinstance(rating, (int, float)) else None,
                        int(count) if isinstance(count, (int, float)) else 0,
                    )
                )
        observed = [rating for _, rating, _ in rows if rating is not None]
        global_mean = statistics.fmean(observed) if observed else 0.0
        maximum_count = max((count for _, _, count in rows), default=0)
        self.quality = {}
        for identifier, rating, count in rows:
            value = rating if rating is not None else global_mean
            bayesian = (count * value + prior_strength * global_mean) / (
                count + prior_strength
            )
            popularity = (
                math.log1p(count) / math.log1p(maximum_count)
                if maximum_count > 0
                else 0.0
            )
            self.quality[identifier] = 0.7 * (bayesian / 5.0) + 0.3 * popularity

    def rerank(
        self,
        ranking: list[str],
        *,
        weight: float,
        rerank_k: int = 50,
    ) -> list[str]:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("quality prior weight must be between zero and one")
        if weight == 0.0:
            return list(ranking)
        head = ranking[:rerank_k]
        count = len(head)
        original_ranks = {identifier: rank for rank, identifier in enumerate(head, 1)}
        scores = {
            identifier: (
                (1.0 if count == 1 else 1.0 - (rank - 1) / max(1, count - 1))
                + weight * self.quality.get(identifier, 0.0)
            )
            for rank, identifier in enumerate(head, 1)
        }
        ordered = sorted(
            head,
            key=lambda identifier: (
                -scores[identifier],
                original_ranks[identifier],
                identifier,
            ),
        )
        return [*ordered, *ranking[rerank_k:]]

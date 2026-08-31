from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ghostlab.retrieval.filters import ConstraintAuthorityResult
from ghostlab.retrieval.gbdt import METADATA_FEATURES, GBDTFeatureStore
from ghostlab.retrieval.multi_route import CandidateEvidence, MergedCandidatePool

SOURCE_AWARE_FEATURES = (
    "route_buying",
    "route_browsing",
    "keyword_member",
    "category_member",
    "vector_member",
    "source_count",
    "keyword_rank_normalized",
    "keyword_reciprocal_rank",
    "keyword_score",
    "keyword_missing",
    "category_rank_normalized",
    "category_reciprocal_rank",
    "category_score",
    "category_missing",
    "vector_rank_normalized",
    "vector_reciprocal_rank",
    "vector_score",
    "vector_missing",
    "aggregate_merge_score",
    "cross_source_agreement",
    "confirmed_constraint_matches",
    "unknown_constraint_count",
    "soft_preference_count",
    "profile_term_match",
)
UNION_FEATURES = (*METADATA_FEATURES, *SOURCE_AWARE_FEATURES)


def _rank_features(rank: int | None, maximum: int) -> tuple[float, float, float]:
    if rank is None:
        return math.nan, math.nan, 1.0
    normalized = 0.0 if maximum <= 1 else (rank - 1) / (maximum - 1)
    return normalized, 1.0 / rank, 0.0


class UnionFeatureStore:
    """Fit/runtime-identical features over the exact three-source candidate union."""

    def __init__(self, base: GBDTFeatureStore) -> None:
        self.base = base

    def all_features(
        self,
        query: str,
        evidence: CandidateEvidence,
        *,
        rank: int,
        count: int,
        route: str,
        source_depths: Mapping[str, int],
        authority: ConstraintAuthorityResult | None = None,
        profile_terms: frozenset[str] = frozenset(),
    ) -> dict[str, float]:
        values = self.base.all_features(
            query, evidence.parent_asin, rank=rank, count=count
        )
        keyword_rank, keyword_rr, keyword_missing = _rank_features(
            evidence.keyword_rank, source_depths.get("keyword", 0)
        )
        category_rank, category_rr, category_missing = _rank_features(
            evidence.category_rank, source_depths.get("category", 0)
        )
        vector_rank, vector_rr, vector_missing = _rank_features(
            evidence.vector_rank, source_depths.get("vector", 0)
        )
        product = self.base.products.get(evidence.parent_asin)
        document_terms = (
            frozenset().union(*product.field_terms) if product is not None else frozenset()
        )
        denominator = max(1, len(profile_terms))
        values.update(
            {
                "route_buying": float(route == "buying"),
                "route_browsing": float(route == "browsing"),
                "keyword_member": float("keyword" in evidence.sources),
                "category_member": float("category" in evidence.sources),
                "vector_member": float("vector" in evidence.sources),
                "source_count": float(len(evidence.sources)),
                "keyword_rank_normalized": keyword_rank,
                "keyword_reciprocal_rank": keyword_rr,
                "keyword_score": (
                    float(evidence.keyword_score)
                    if evidence.keyword_score is not None
                    else math.nan
                ),
                "keyword_missing": keyword_missing,
                "category_rank_normalized": category_rank,
                "category_reciprocal_rank": category_rr,
                "category_score": (
                    float(evidence.category_score)
                    if evidence.category_score is not None
                    else math.nan
                ),
                "category_missing": category_missing,
                "vector_rank_normalized": vector_rank,
                "vector_reciprocal_rank": vector_rr,
                "vector_score": (
                    float(evidence.vector_score)
                    if evidence.vector_score is not None
                    else math.nan
                ),
                "vector_missing": vector_missing,
                "aggregate_merge_score": float(evidence.aggregate_score),
                "cross_source_agreement": len(evidence.sources) / 3.0,
                "confirmed_constraint_matches": float(
                    authority.confirmed_match_count.get(evidence.parent_asin, 0)
                    if authority is not None
                    else 0
                ),
                "unknown_constraint_count": float(
                    authority.unknown_count.get(evidence.parent_asin, 0)
                    if authority is not None
                    else 0
                ),
                "soft_preference_count": float(
                    authority.soft_preference_count.get(evidence.parent_asin, 0)
                    if authority is not None
                    else 0
                ),
                "profile_term_match": len(profile_terms & document_terms) / denominator,
            }
        )
        return values

    def matrix(
        self,
        query: str,
        pool: MergedCandidatePool,
        feature_names: Sequence[str],
        *,
        authority: ConstraintAuthorityResult | None = None,
        profile_terms: frozenset[str] = frozenset(),
    ) -> NDArray[np.float64]:
        unknown = set(feature_names) - set(UNION_FEATURES)
        if unknown:
            raise ValueError(f"unknown union features: {sorted(unknown)}")
        count = len(pool.candidates)
        if not count:
            return np.empty((0, len(feature_names)), dtype=np.float64)
        depths = {
            source: max(
                (
                    getattr(item, f"{source}_rank") or 0
                    for item in pool.candidates
                ),
                default=0,
            )
            for source in ("keyword", "category", "vector")
        }
        return np.asarray(
            [
                [features[name] for name in feature_names]
                for rank, evidence in enumerate(pool.candidates, start=1)
                for features in (
                    self.all_features(
                        query,
                        evidence,
                        rank=rank,
                        count=count,
                        route=pool.route,
                        source_depths=depths,
                        authority=authority,
                        profile_terms=profile_terms,
                    ),
                )
            ],
            dtype=np.float64,
        )


__all__ = ["SOURCE_AWARE_FEATURES", "UNION_FEATURES", "UnionFeatureStore"]

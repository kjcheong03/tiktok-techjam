from __future__ import annotations

import unittest

from ghostlab.policy.models import RankedCandidate, RankedCandidates
from ghostlab.retrieval.fusion import fuse_rankings, jaccard_at, weighted_fuse_ids


def ranking(route: str, identifiers: list[str]) -> RankedCandidates:
    return RankedCandidates(
        route=route,
        requested_k=10,
        elapsed_ms=1.0,
        items=tuple(
            RankedCandidate(
                parent_asin=value,
                route=route,
                rank=index,
                normalized_score=1.0 - (index - 1) / max(1, len(identifiers) - 1),
            )
            for index, value in enumerate(identifiers, start=1)
        ),
    )


class RetrievalContractTest(unittest.TestCase):
    def test_weighted_fusion_is_deterministic_and_unique(self) -> None:
        result = fuse_rankings(
            ranking("keyword", ["a", "b", "c"]),
            ranking("dense", ["b", "d", "a"]),
            limit=4,
        )
        self.assertEqual(len({item.parent_asin for item in result.items}), 4)
        self.assertEqual([item.rank for item in result.items], [1, 2, 3, 4])

    def test_overlap_missing_and_disjoint_semantics(self) -> None:
        self.assertIsNone(jaccard_at([], ["a"], 10))
        self.assertEqual(jaccard_at(["a"], ["b"], 10), 0.0)

    def test_duplicate_ranked_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ranking("keyword", ["a", "a"])

    def test_keyword_heavy_fusion_retains_sparse_leader(self) -> None:
        result = weighted_fuse_ids(["a", "b", "c"], ["d", "b", "a"], limit=4)
        self.assertEqual(result[0], "a")
        self.assertEqual(len(result), len(set(result)))


if __name__ == "__main__":
    unittest.main()

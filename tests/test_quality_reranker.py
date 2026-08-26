from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.retrieval.quality import CatalogQualityReranker


class CatalogQualityRerankerTests(unittest.TestCase):
    def test_high_confidence_quality_is_a_soft_boost(self) -> None:
        products = [
            {"parent_asin": "low", "average_rating": 2.0, "rating_number": 1000},
            {"parent_asin": "high", "average_rating": 5.0, "rating_number": 1000},
            {"parent_asin": "third", "average_rating": 2.0, "rating_number": 1},
            {"parent_asin": "last", "average_rating": 2.0, "rating_number": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in products))
            reranker = CatalogQualityReranker(path, prior_strength=1.0)
            result = reranker.rerank(
                ["low", "high", "third", "last"], weight=1.0, rerank_k=4
            )
        self.assertEqual(result[0], "high")

    def test_zero_weight_preserves_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                json.dumps(
                    {"parent_asin": "a", "average_rating": 5.0, "rating_number": 1}
                )
                + "\n"
            )
            reranker = CatalogQualityReranker(path)
            self.assertEqual(reranker.rerank(["a"], weight=0.0), ["a"])


if __name__ == "__main__":
    unittest.main()

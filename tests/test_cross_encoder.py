from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ghostlab.retrieval.cross_encoder import (
    PASSAGE_SCHEMA_VERSION,
    CrossEncoderReranker,
    blend_ranking,
    product_passage,
)


class FakeScorer:
    def predict(self, inputs, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return np.asarray(
            [1.0 if "preferred" in passage else 0.0 for _, passage in inputs],
            dtype=np.float32,
        )


class CrossEncoderTest(unittest.TestCase):
    def test_product_passage_has_field_labels(self) -> None:
        passage = product_passage(
            {
                "title": "Boot",
                "categories": ["Shoes"],
                "price": 49.0,
                "store": "Example",
                "features": ["Waterproof"],
            }
        )
        self.assertEqual(PASSAGE_SCHEMA_VERSION, "catalog_fields_v2")
        self.assertIn("title: Boot", passage)
        self.assertIn("price: 49.0", passage)
        self.assertIn("store: Example", passage)
        self.assertIn("features: Waterproof", passage)

    def test_product_passage_bounds_each_field(self) -> None:
        passage = product_passage(
            {"title": " ".join(f"word{index}" for index in range(100))}
        )
        self.assertIn("word31", passage)
        self.assertNotIn("word32", passage)

    def test_blend_preserves_base_at_zero_and_uses_scores_at_one(self) -> None:
        ranking = ["a", "b", "c"]
        self.assertEqual(blend_ranking(ranking, [0.0, 1.0, 0.5], weight=0.0), ranking)
        self.assertEqual(
            blend_ranking(ranking, [0.0, 1.0, 0.5], weight=1.0),
            ["b", "c", "a"],
        )

    def test_reranker_scores_only_the_bounded_head_and_caches(self) -> None:
        products = [
            {"parent_asin": "a", "title": "ordinary"},
            {"parent_asin": "b", "title": "preferred"},
            {"parent_asin": "c", "title": "tail"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(item) + "\n" for item in products),
                encoding="utf-8",
            )
            reranker = CrossEncoderReranker(
                catalog,
                model_name="fake",
                revision="fake",
                cache_folder=directory,
                scorer=FakeScorer(),
            )
            result = reranker.rerank("query", ["a", "b", "c"], rerank_k=2, weight=1.0)
            self.assertEqual(result, ["b", "a", "c"])
            reranker.rerank("query", ["a", "b", "c"], rerank_k=2, weight=1.0)
            self.assertEqual(reranker.cache_hits, 2)
            self.assertEqual(reranker.cache_misses, 2)
            self.assertEqual(reranker.score_calls, 1)


if __name__ == "__main__":
    unittest.main()

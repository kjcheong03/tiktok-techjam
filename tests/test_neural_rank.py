from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ghostlab.retrieval.gbdt import METADATA_FEATURES, GBDTFeatureStore
from ghostlab.retrieval.neural_rank import (
    NEURAL_METADATA_FEATURES,
    PASSAGE_SCHEMA_VERSION,
    NeuralGBDTFeatureStore,
    NeuralScoreCache,
    PinnedCrossEncoderScorer,
    product_passage,
    score_cache_identity,
    write_score_cache,
)


class FakeScorer:
    def predict(self, inputs, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return np.asarray(
            [1.0 if "preferred" in passage else 0.25 for _, passage in inputs],
            dtype=np.float32,
        )


class NeuralRankTest(unittest.TestCase):
    def test_passage_schema_and_budgets_match_pinned_cross_encoder(self) -> None:
        passage = product_passage(
            {
                "title": " ".join(f"word{index}" for index in range(40)),
                "categories": ["Shoes"],
                "features": ["Waterproof"],
            }
        )
        self.assertEqual(PASSAGE_SCHEMA_VERSION, "catalog_fields_v2")
        self.assertIn("word31", passage)
        self.assertNotIn("word32", passage)
        self.assertIn("category: Shoes", passage)

    def test_score_cache_round_trip_and_feature_missingness(self) -> None:
        products = [
            {"parent_asin": "a", "title": "ordinary"},
            {"parent_asin": "b", "title": "preferred"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(item) + "\n" for item in products),
                encoding="utf-8",
            )
            catalog_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
            identity = score_cache_identity(catalog_hash)
            scorer = PinnedCrossEncoderScorer(catalog, root, scorer=FakeScorer())
            cache_path = root / "scores.jsonl"
            write_score_cache(cache_path, identity, [("query", "a")], scorer)
            cache = NeuralScoreCache(cache_path, identity)
            base = GBDTFeatureStore(catalog)
            store = NeuralGBDTFeatureStore(base, cache=cache)
            matrix = store.matrix("query", ["a", "b"], NEURAL_METADATA_FEATURES)
            self.assertEqual(matrix.shape, (2, len(METADATA_FEATURES) + 2))
            self.assertEqual(matrix[0, -2:].tolist(), [0.25, 0.0])
            self.assertTrue(np.isnan(matrix[1, -2]))
            self.assertEqual(matrix[1, -1], 1.0)

    def test_live_scorer_fills_cache_misses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                json.dumps({"parent_asin": "a", "title": "preferred"}) + "\n",
                encoding="utf-8",
            )
            live = PinnedCrossEncoderScorer(catalog, directory, scorer=FakeScorer())
            store = NeuralGBDTFeatureStore(GBDTFeatureStore(catalog), live_scorer=live)
            matrix = store.matrix("query", ["a"], NEURAL_METADATA_FEATURES)
            self.assertEqual(matrix[0, -2:].tolist(), [1.0, 0.0])
            self.assertEqual(live.scored_pairs, 1)


if __name__ == "__main__":
    unittest.main()

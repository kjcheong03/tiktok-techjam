from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ghostlab.retrieval.gbdt import (
    FEATURE_SETS,
    GBDTFeatureStore,
    LambdaMARTModel,
    LambdaMARTReranker,
    fit_lambdamart,
)


class GBDTRerankerTests(unittest.TestCase):
    def _catalog(self, directory: str) -> Path:
        path = Path(directory) / "catalog.jsonl"
        rows = [
            {
                "parent_asin": "plain",
                "title": "plain item",
                "average_rating": None,
            },
            {
                "parent_asin": "match",
                "title": "waterproof shell",
                "features": ["waterproof"],
                "average_rating": 4.8,
                "rating_number": 100,
            },
            {"parent_asin": "tail", "title": "tail item"},
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_missing_values_have_native_and_indicator_representations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GBDTFeatureStore(self._catalog(directory))
            missing = store.all_features("waterproof", "plain", rank=1, count=3)
            observed = store.all_features("waterproof", "match", rank=2, count=3)
        self.assertTrue(math.isnan(missing["average_rating"]))
        self.assertEqual(missing["average_rating_missing"], 1.0)
        self.assertEqual(missing["feature_overlap"], 0.0)
        self.assertEqual(missing["feature_overlap_missing"], 1.0)
        self.assertEqual(observed["average_rating_missing"], 0.0)
        self.assertGreater(observed["feature_overlap"], 0.0)

    def test_fit_is_deterministic_and_model_round_trips(self) -> None:
        features = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        kwargs = {
            "candidate_id": "toy",
            "feature_names": ("first", "second"),
            "max_depth": 1,
            "num_leaves": 2,
            "learning_rate": 0.1,
            "max_rounds": 2,
            "early_stopping_rounds": 3,
        }
        first = fit_lambdamart(features, labels, [2, 2], **kwargs)
        second = fit_lambdamart(features, labels, [2, 2], **kwargs)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            first.save(path)
            loaded = LambdaMARTModel.load(path)
        self.assertEqual(loaded, first)

    def test_reranker_promotes_learned_match_and_preserves_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GBDTFeatureStore(self._catalog(directory))
            names = FEATURE_SETS["lexical"]
            ranking = ["plain", "match"]
            positive = store.matrix("waterproof", ranking, names)
            model = fit_lambdamart(
                np.vstack([positive] * 50),
                np.tile(np.asarray([0, 1], dtype=np.int64), 50),
                [2] * 50,
                candidate_id="toy",
                feature_names=names,
                max_depth=2,
                num_leaves=4,
                learning_rate=0.1,
                max_rounds=3,
                early_stopping_rounds=3,
            )
            reranked = LambdaMARTReranker(store, model).rerank(
                "waterproof", ["plain", "match", "tail"], rerank_k=2
            )
        self.assertEqual(reranked[-1], "tail")
        self.assertEqual(reranked[0], "match")

    def test_invalid_group_boundaries_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not align"):
            fit_lambdamart(
                np.zeros((2, 1)),
                np.zeros(2, dtype=np.int64),
                [3],
                candidate_id="bad",
                feature_names=("rank",),
                max_depth=1,
                num_leaves=2,
                learning_rate=0.1,
                max_rounds=2,
                early_stopping_rounds=1,
            )


if __name__ == "__main__":
    unittest.main()

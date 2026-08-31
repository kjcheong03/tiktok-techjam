from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ghostlab.retrieval.learned import (
    CandidateFeatureStore,
    LearnedLinearReranker,
    PairwiseExample,
    fit_pairwise_linear,
)


class LearnedRerankerTests(unittest.TestCase):
    def test_pairwise_fit_learns_positive_residual(self) -> None:
        examples = [PairwiseExample(-0.2, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))]
        model = fit_pairwise_linear(examples, l2=0.01)
        self.assertGreater(model.weights[0], 0.2)

    def test_model_can_promote_feature_match(self) -> None:
        products = [
            {"parent_asin": "plain", "title": "plain item"},
            {"parent_asin": "match", "features": ["waterproof"]},
            {"parent_asin": "last", "title": "last item"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in products))
            store = CandidateFeatureStore(path)
            model = fit_pairwise_linear(
                [
                    PairwiseExample(
                        -0.5,
                        tuple(
                            left - right
                            for left, right in zip(
                                store.features("waterproof", "match"),
                                store.features("waterproof", "plain"),
                                strict=True,
                            )
                        ),
                    )
                ],
                l2=0.01,
            )
            result = LearnedLinearReranker(store, model).rerank(
                "waterproof", ["plain", "match", "last"], rerank_k=3
            )
        self.assertEqual(result[0], "match")

    def test_disabled_features_remain_zero(self) -> None:
        example = PairwiseExample(-0.2, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        model = fit_pairwise_linear([example], enabled_features=("feature_overlap",))
        self.assertGreater(model.weights[2], 0.0)
        self.assertEqual(
            tuple(weight for index, weight in enumerate(model.weights) if index != 2),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

    def test_runtime_store_skips_disabled_catalog_fields(self) -> None:
        product = {
            "parent_asin": "item",
            "title": "unused title",
            "features": ["waterproof"],
            "average_rating": 4.5,
            "rating_number": 20,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps(product) + "\n", encoding="utf-8")
            store = CandidateFeatureStore(
                path, enabled_features=("feature_overlap", "catalog_quality")
            )
            values = store.features("unused waterproof", "item")
        self.assertEqual(values[0], 0.0)
        self.assertGreater(values[2], 0.0)
        self.assertGreater(values[6], 0.0)


if __name__ == "__main__":
    unittest.main()

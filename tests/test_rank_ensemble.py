import tempfile
import unittest
from pathlib import Path

import numpy as np

from ghostlab.retrieval.ensemble import (
    RankEnsembleAsset,
    aggregate_rankings,
    combine_model_scores,
    fit_rank_stack_weights,
)


class RankEnsembleTests(unittest.TestCase):
    def test_rank_aggregation_is_deterministic_and_handles_missing_items(self) -> None:
        rankings = [["a", "b", "c"], ["b", "a", "d"]]
        first = aggregate_rankings(rankings, method="mean_rank")
        second = aggregate_rankings(rankings, method="mean_rank")
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"a", "b", "c", "d"})
        self.assertEqual(first[:2], ["a", "b"])

    def test_score_ensemble_standardizes_incompatible_head_scales(self) -> None:
        heads = np.asarray([[1.0, 0.0], [1000.0, 999.0]])
        combined = combine_model_scores(
            heads, method="standardized_score", weights=(0.5, 0.5)
        )
        self.assertGreater(combined[0], combined[1])

    def test_fold_local_stacker_selects_complementary_head(self) -> None:
        labels = np.asarray([1, 0, 0, 1], dtype=np.int64)
        heads = np.asarray(
            [
                [2.0, 0.0, 0.0, 2.0],
                [0.0, 2.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        asset = fit_rank_stack_weights(heads, labels, [2, 2], [1, 1], grid_step=0.25)
        self.assertEqual(asset.technique_id, "fusion.rank_stack.v1")
        self.assertGreater(asset.weights[0], asset.weights[1])

    def test_invalid_weight_vectors_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_rankings([["a"], ["b"]], weights=(1.0,))
        with self.assertRaises(ValueError):
            combine_model_scores(
                np.ones((2, 2)), method="mean_rank", weights=(0.0, 0.0)
            )

    def test_asset_round_trip_rejects_unsafe_model_paths(self) -> None:
        asset = RankEnsembleAsset(
            technique_id="ranking.fold_ensemble.v1",
            aggregation="standardized_score",
            model_assets=("artifacts/models/first.json",),
            weights=(1.0,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.json"
            asset.save(path)
            loaded = RankEnsembleAsset.load(path)
        self.assertEqual(asset, loaded)
        with self.assertRaises(ValueError):
            RankEnsembleAsset(
                technique_id="ranking.fold_ensemble.v1",
                aggregation="standardized_score",
                model_assets=("../protected/model.json",),
                weights=(1.0,),
            )


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Wave2RankingReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (ROOT / "artifacts/reports/w2_ranking_v1.json").read_text()
        )

    def test_report_is_outer_oof_and_protected_split_stays_sealed(self) -> None:
        self.assertFalse(self.report["protected_holdout_accessed"])
        self.assertIn("outer-fold OOF", self.report["evidence_label"])
        for candidate in self.report["candidates"].values():
            self.assertEqual(candidate["oof_metrics"]["sample_count"], 150)
            self.assertEqual(len(candidate["folds"]), 5)

    def test_exact_historical_ndcg_control_is_reproduced(self) -> None:
        metrics = self.report["candidates"]["ndcg_at_10_control"]["oof_metrics"]
        self.assertEqual(metrics["recommended_technical_score"], 0.861417)
        self.assertEqual(metrics["hit_rate_at_10"], 0.973333)

    def test_leader_is_parked_and_not_misreported_as_champion(self) -> None:
        leader = self.report["oof_leader"]
        self.assertEqual(leader["candidate_id"], "equal_standardized_scores")
        self.assertEqual(leader["promotion_status"], "research_only_not_promoted")
        paired = self.report["candidates"][leader["candidate_id"]][
            "paired_vs_ndcg_control"
        ]
        self.assertGreater(paired["mean_paired_session_reward_delta"], 0.0)
        self.assertLess(paired["paired_bootstrap_95_interval"][0], 0.0)

    def test_runtime_assets_contain_no_target_features(self) -> None:
        forbidden = {"target", "future_hit_turn", "scenario", "fold_id", "reward"}
        model_dir = ROOT / "artifacts/models/w2_ranking_v1"
        for path in model_dir.glob("*_control.json"):
            model = json.loads(path.read_text())
            self.assertTrue(forbidden.isdisjoint(model["feature_names"]))
        for path in model_dir.glob("*_v1.json"):
            model = json.loads(path.read_text())
            self.assertTrue(forbidden.isdisjoint(model["feature_names"]))

    def test_reserved_ids_have_explicit_off_by_default_bindings(self) -> None:
        catalog = json.loads(
            (ROOT / "configs/techniques/w2_ranking_v1.json").read_text()
        )
        identifiers = {item["id"] for item in catalog["techniques"]}
        self.assertEqual(catalog["default_state"], "off")
        self.assertTrue(
            {
                "ranking.reward_lambdamart.v1",
                "ranking.turn_aware_lambdamart.v1",
                "ranking.fold_ensemble.v1",
                "fusion.rank_stack.v1",
            }.issubset(identifiers)
        )


if __name__ == "__main__":
    unittest.main()

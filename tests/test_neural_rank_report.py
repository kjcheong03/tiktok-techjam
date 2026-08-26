from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/reports/neural_rank_interaction_v1.json"


class NeuralRankReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def test_interaction_is_parked_without_deployable_refit(self) -> None:
        self.assertFalse(self.report["holdout_accessed"])
        self.assertFalse(self.report["f3_available"])
        self.assertEqual(self.report["selection"]["decision"], "PARK_INTERACTION")
        self.assertFalse(self.report["selection"]["deployable_model_written"])
        self.assertFalse(self.report["fit_audit"]["all_development_refit_performed"])
        self.assertTrue(self.report["fit_audit"]["all_learned_tree_fits_inside_folds"])

    def test_complete_oof_and_paired_evidence(self) -> None:
        sessions = self.report["candidate"]["oof_sessions"]
        identifiers = [str(item["sample_id"]) for item in sessions]
        self.assertEqual(len(identifiers), 150)
        self.assertEqual(len(set(identifiers)), 150)
        self.assertEqual(len(self.report["candidate"]["folds"]), 5)
        self.assertEqual(
            self.report["candidate"]["paired_vs_audited_gbdt"]["resamples"],
            10000,
        )
        self.assertLess(
            self.report["candidate"]["oof_metrics"]["recommended_technical_score"],
            self.report["control"]["oof_metrics"]["recommended_technical_score"],
        )

    def test_pinned_cache_is_complete_and_feature_is_used(self) -> None:
        cache = self.report["score_cache"]
        self.assertTrue(cache["complete_for_all_frozen_trajectory_top50_pairs"])
        self.assertEqual(cache["row_count"], 74300)
        self.assertEqual(
            cache["identity"]["model_revision"],
            "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        )
        self.assertEqual(
            cache["identity"]["passage_schema_version"], "catalog_fields_v2"
        )
        importance = self.report["candidate"][
            "feature_importance_split_count_across_outer_models"
        ]
        self.assertGreater(importance["cross_encoder_score"], 0)
        self.assertEqual(importance["cross_encoder_score_missing"], 0)

    def test_runtime_and_hashes(self) -> None:
        self.assertEqual(self.report["performance"]["failure_count"], 0)
        self.assertEqual(self.report["performance"]["external_calls_per_turn"], 0)
        self.assertFalse(self.report["performance"]["passed"])
        paths = {
            "ghostlab/retrieval/gbdt.py": ROOT / "ghostlab/retrieval/gbdt.py",
            "ghostlab/retrieval/neural_rank.py": ROOT
            / "ghostlab/retrieval/neural_rank.py",
            "scripts/run_gbdt_reranker.py": ROOT / "scripts/run_gbdt_reranker.py",
            "scripts/run_neural_rank_interaction.py": ROOT
            / "scripts/run_neural_rank_interaction.py",
            "scripts/measure_neural_rank_runtime.py": ROOT
            / "scripts/measure_neural_rank_runtime.py",
        }
        for key, path in paths.items():
            self.assertEqual(
                self.report["code_hashes"][key],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()

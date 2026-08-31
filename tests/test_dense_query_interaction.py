from __future__ import annotations

import json
import unittest
from pathlib import Path

from ghostlab.retrieval.dense import sha256_file


class DenseQueryInteractionReportTest(unittest.TestCase):
    def test_retrieval_report_is_current_and_cache_only(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (root / "artifacts/reports/dense_query_interaction_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(report["holdout_accessed"])
        self.assertFalse(report["model_search_performed"])
        self.assertFalse(report["index_build_performed"])
        self.assertEqual(report["query_record_count"], 1350)
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(
            report["gate"]["selected_candidate"], "negation_safe_structured"
        )
        paths = {
            "manifest_sha256": "configs/experiments/dense_query_interaction_v1.json",
            "query_code_sha256": "ghostlab/retrieval/query.py",
            "runner_code_sha256": "scripts/run_dense_query_interaction.py",
        }
        for key, relative in paths.items():
            self.assertEqual(report["hashes"][key], sha256_file(root / relative))

    def test_end_to_end_report_uses_champion_oof_control(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (root / "artifacts/reports/dense_query_e2e_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(report["holdout_accessed"])
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(
            report["baseline_metrics"]["recommended_technical_score"], 0.817649
        )
        self.assertEqual(
            report["candidate_metrics"]["recommended_technical_score"], 0.835125
        )
        self.assertEqual(report["decision"], "PROMOTE_INTERACTION")
        self.assertTrue(report["gate"]["passed"])
        paths = {
            "manifest_sha256": "configs/experiments/dense_query_e2e_v1.json",
            "retrieval_report_sha256": "artifacts/reports/dense_query_interaction_v1.json",
            "runner_code_sha256": "scripts/run_dense_query_e2e.py",
        }
        for key, relative in paths.items():
            self.assertEqual(report["hashes"][key], sha256_file(root / relative))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from server import count_visualizable_runs, discover_reports


class ReportDiscoveryTests(unittest.TestCase):
    def test_counts_direct_evaluator_report(self) -> None:
        payload = {
            "sample_count": 2,
            "hit_rate_at_10": 0.5,
            "mrr": 0.25,
            "sessions": [],
        }
        self.assertEqual(count_visualizable_runs(payload), 1)

    def test_counts_nested_and_multi_run_reports(self) -> None:
        nested = {"experiment_id": "candidate", "metrics": {"mrr": 0.7}}
        multi = {
            "baseline": {"mrr": 0.2},
            "candidate": {"recommended_technical_score": 0.8},
        }
        self.assertEqual(count_visualizable_runs(nested), 1)
        self.assertEqual(count_visualizable_runs(multi), 2)

    def test_counts_campaign_records(self) -> None:
        payload = {
            "schema_version": 1,
            "records": [
                {"ordinal": 1, "metrics": {"mrr": 0.3}},
                {"ordinal": 2, "metrics": {"hit_rate_at_10": 0.9}},
            ],
        }
        self.assertEqual(count_visualizable_runs(payload), 2)

    def test_ignores_non_metric_json(self) -> None:
        self.assertEqual(count_visualizable_runs({"status": "ok", "records": []}), 0)

    def test_discovers_repository_reports(self) -> None:
        reports = discover_reports()
        paths = {str(report["path"]) for report in reports}
        self.assertIn(
            "artifacts/reports/unified_champion_verification_v1.json", paths
        )
        self.assertTrue(all(int(report["run_count"]) > 0 for report in reports))


if __name__ == "__main__":
    unittest.main()

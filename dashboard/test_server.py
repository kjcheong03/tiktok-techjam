from __future__ import annotations

import unittest

from server import (
    _select_comparison_systems,
    count_visualizable_runs,
    discover_models,
    discover_reports,
)


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

    def test_counts_fair_system_comparison(self) -> None:
        payload = {
            "comparison_semantics": {"same_ground": True},
            "systems": [
                {"system_id": "A", "metrics": {"mrr": 0.3}},
                {"system_id": "B", "metrics": {"hit_rate_at_10": 0.9}},
                {"system_id": "metadata-only"},
            ],
        }
        self.assertEqual(count_visualizable_runs(payload), 2)

    def test_ignores_non_metric_json(self) -> None:
        self.assertEqual(count_visualizable_runs({"status": "ok", "records": []}), 0)

    def test_discovers_repository_reports(self) -> None:
        reports = discover_reports()
        paths = {str(report["path"]) for report in reports}
        self.assertIn("artifacts/reports/unified_champion_verification_v1.json", paths)
        self.assertTrue(all(int(report["run_count"]) > 0 for report in reports))

    def test_discovers_only_four_stable_model_slots(self) -> None:
        models = discover_models()
        self.assertEqual([model["model_id"] for model in models], ["A", "B", "C", "D"])
        self.assertEqual(
            [model["label"] for model in models],
            [
                "A: BM25",
                "B: BM25 + teammate State V2",
                "C: adaptive control",
                "D: frozen GhostLab champion / challenger",
            ],
        )
        self.assertEqual(len({model["model_id"] for model in models}), 4)
        self.assertEqual(sum(bool(model["featured"]) for model in models), 1)

    def test_promoted_challenger_occupies_one_d_slot(self) -> None:
        payload = {
            "selected_system_id": "champion_latest",
            "systems": [
                {"system_id": "A_bm25"},
                {"system_id": "B_state_v2"},
                {"system_id": "C_adaptive"},
                {
                    "system_id": "D1_candidate_42",
                    "candidate_id": "candidate_42",
                    "role": "ghostlab_challenger",
                },
                {
                    "system_id": "champion_latest",
                    "candidate_id": "candidate_42",
                    "role": "ghostlab_champion",
                },
            ],
        }
        selected = _select_comparison_systems(payload)
        self.assertEqual(set(selected), {"A", "B", "C", "D"})
        self.assertEqual(selected["D"]["system_id"], "champion_latest")


if __name__ == "__main__":
    unittest.main()

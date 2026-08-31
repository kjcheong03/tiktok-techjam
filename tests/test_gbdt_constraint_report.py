from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/gbdt_constraint_interaction_v1.json"


class GBDTConstraintReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_holdout_and_control_are_frozen(self) -> None:
        self.assertFalse(self.report["holdout_accessed"])
        self.assertEqual(
            self.report["parent_commit"],
            "cbfd7d5dd595c5637608ba28f46f57777c7e153e",
        )
        self.assertTrue(self.report["matched_control"]["exactly_reproduced"])
        self.assertEqual(
            self.report["matched_control"]["oof_metrics"][
                "recommended_technical_score"
            ],
            0.861417,
        )

    def test_candidate_sessions_are_unique_and_fold_disjoint(self) -> None:
        expected = set(
            json.loads((ROOT / "configs/splits/nested_v1.json").read_text())[
                "adaptive_sample_ids"
            ]
        )
        sessions = self.report["candidate"]["oof_sessions"]
        identifiers = [str(session["sample_id"]) for session in sessions]
        self.assertEqual(len(identifiers), 150)
        self.assertEqual(set(identifiers), expected)
        for fold in self.report["candidate"]["folds"]:
            self.assertFalse(
                set(fold["outer_training_ids"]) & set(fold["outer_validation_ids"])
            )

    def test_promotion_gates_and_backward_ablation_are_executable(self) -> None:
        self.assertTrue(self.report["promotion"]["all_gates_passed"])
        self.assertTrue(all(self.report["promotion"]["gate_results"].values()))
        self.assertEqual(
            self.report["backward_ablation"]["metrics"],
            self.report["matched_control"]["oof_metrics"],
        )
        self.assertGreaterEqual(
            sum(delta >= 0 for delta in self.report["candidate"]["fold_score_deltas"]),
            4,
        )
        self.assertGreaterEqual(
            min(self.report["candidate"]["scenario_reward_deltas"].values()),
            -0.005,
        )

    def test_superseded_artifact_provenance_remains_immutable(self) -> None:
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        self.assertEqual(
            digest(ROOT / self.report["manifest_path"]),
            self.report["manifest_sha256"],
        )
        refit = self.report["all_development_refit"]
        self.assertEqual(digest(ROOT / refit["model_path"]), refit["model_sha256"])
        self.assertTrue(self.report["runtime"]["contention_affected"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/gbdt_constraint_interaction_v2.json"


class GBDTConstraintV2ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_amendment_holdout_and_supersession_are_frozen(self) -> None:
        self.assertFalse(self.report["holdout_accessed"])
        self.assertEqual(
            self.report["supersedes_report"],
            "artifacts/reports/gbdt_constraint_interaction_v1.json",
        )
        amendment = ROOT / self.report["amendment_path"]
        self.assertEqual(
            hashlib.sha256(amendment.read_bytes()).hexdigest(),
            self.report["amendment_sha256"],
        )

    def test_control_and_grouped_sessions_are_exact(self) -> None:
        self.assertTrue(self.report["matched_control"]["exactly_reproduced"])
        self.assertEqual(
            self.report["matched_control"]["oof_metrics"][
                "recommended_technical_score"
            ],
            0.861417,
        )
        expected = set(
            json.loads((ROOT / "configs/splits/nested_v1.json").read_text())[
                "adaptive_sample_ids"
            ]
        )
        identifiers = [
            str(session["sample_id"])
            for session in self.report["candidate"]["oof_sessions"]
        ]
        self.assertEqual(len(identifiers), 150)
        self.assertEqual(set(identifiers), expected)

    def test_robustness_failures_are_not_promoted(self) -> None:
        promotion = self.report["promotion"]
        self.assertFalse(promotion["all_gates_passed"])
        self.assertEqual(promotion["decision"], "PARK_INTERACTION")
        self.assertFalse(promotion["gate_results"]["minimum_hit_rate_delta"])
        self.assertFalse(promotion["gate_results"]["minimum_each_scenario_score_delta"])
        lower, upper = self.report["candidate"]["paired_vs_matched_control"][
            "paired_bootstrap_95_interval"
        ]
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)

    def test_model_and_code_hashes_match(self) -> None:
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        for relative, expected in self.report["code_hashes"].items():
            self.assertEqual(digest(ROOT / relative), expected)
        model = self.report["all_development_refit"]
        self.assertEqual(digest(ROOT / model["model_path"]), model["model_sha256"])
        self.assertFalse(self.report["runtime"]["contention_affected"])


if __name__ == "__main__":
    unittest.main()

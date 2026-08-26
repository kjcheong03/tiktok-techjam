from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/challenger_cross_encoder_v1.json"


class CrossEncoderReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_holdout_and_decision(self) -> None:
        self.assertFalse(self.report["holdout_accessed"])
        self.assertEqual(self.report["decision"]["status"], "PARKED_STANDALONE")
        self.assertEqual(self.report["total_runtime_failures"], 0)

    def test_stitched_outer_sessions_are_complete_and_unique(self) -> None:
        identifiers = [
            str(item["sample_id"]) for item in self.report["nested_selected_sessions"]
        ]
        self.assertEqual(len(identifiers), 150)
        self.assertEqual(len(set(identifiers)), 150)

    def test_manifest_and_runtime_hashes_match(self) -> None:
        expected = self.report["hashes"]
        paths = {
            "catalog_sha256": ROOT / "data/catalog.jsonl",
            "public_set_sha256": ROOT / "data/public_set.jsonl",
            "split_sha256": ROOT / "configs/splits/nested_v1.json",
            "manifest_sha256": ROOT
            / "configs/experiments/challenger_cross_encoder_v1.json",
            "implementation_sha256": ROOT / "ghostlab/retrieval/cross_encoder.py",
            "runner_sha256": ROOT / "scripts/run_cross_encoder_challenger.py",
        }
        for key, path in paths.items():
            if key == "catalog_sha256" and not path.exists():
                continue
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), expected[key]
            )

    def test_candidate_set_was_predeclared_and_champion_was_not_selected(self) -> None:
        self.assertTrue(
            self.report["candidate_set_predeclared_before_neural_completion"]
        )
        self.assertNotIn(
            "control_linear_champion_oof",
            {item["selected"] for item in self.report["fold_selections"]},
        )
        self.assertLess(
            self.report["nested_selected_metrics"]["recommended_technical_score"],
            self.report["variants"]["control_linear_champion_oof"]["metrics"][
                "recommended_technical_score"
            ],
        )


if __name__ == "__main__":
    unittest.main()

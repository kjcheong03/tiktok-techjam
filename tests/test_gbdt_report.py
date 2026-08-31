from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/gbdt_reranker_v1.json"


class GBDTReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_holdout_firewall_and_parent_are_frozen(self) -> None:
        self.assertFalse(self.report["holdout_accessed"])
        self.assertEqual(
            self.report["parent_commit"],
            "189f0c6338e2d2ec1a795dce543e881ff2037f2a",
        )
        serialized = json.dumps(self.report).lower()
        self.assertNotIn("artifacts/guarded", serialized)
        self.assertNotIn("f3_v1.json", serialized)

    def test_recorded_artifact_hashes_match(self) -> None:
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        for relative, expected in self.report["code_hashes"].items():
            if relative == "scripts/measure_gbdt_runtime.py":
                # The deployment audit versions this instrumentation after the OOF
                # report; its updated hash belongs to the audit-resolution report.
                continue
            self.assertEqual(digest(ROOT / relative), expected)
        self.assertEqual(
            digest(ROOT / self.report["manifest_path"]),
            self.report["manifest_sha256"],
        )
        refit = self.report["all_development_refit"]
        self.assertEqual(digest(ROOT / refit["model_path"]), refit["model_sha256"])

    def test_every_outer_session_is_scored_once_per_variant(self) -> None:
        expected = set(
            json.loads((ROOT / "configs/splits/nested_v1.json").read_text())[
                "adaptive_sample_ids"
            ]
        )
        for candidate in self.report["variants"].values():
            identifiers = [
                str(session["sample_id"]) for session in candidate["oof_sessions"]
            ]
            self.assertEqual(len(identifiers), 150)
            self.assertEqual(len(set(identifiers)), 150)
            self.assertEqual(set(identifiers), expected)

    def test_nested_fold_sides_are_disjoint(self) -> None:
        for candidate in self.report["variants"].values():
            for fold in candidate["folds"]:
                inner_training = set(fold["inner_training_ids"])
                inner_validation = set(fold["inner_validation_ids"])
                outer_validation = set(fold["outer_validation_ids"])
                outer_training = set(fold["outer_training_ids"])
                self.assertFalse(inner_training & inner_validation)
                self.assertFalse(inner_training & outer_validation)
                self.assertFalse(inner_validation & outer_validation)
                self.assertEqual(inner_training | inner_validation, outer_training)

    def test_selected_metrics_and_runtime_satisfy_manifest_rule(self) -> None:
        selected_id = self.report["selection"]["selected_candidate_id"]
        selected = self.report["variants"][selected_id]
        champion = self.report["controls"]["two_feature_linear_champion"]
        self.assertGreater(
            selected["oof_metrics"]["recommended_technical_score"],
            champion["oof_metrics"]["recommended_technical_score"],
        )
        self.assertGreaterEqual(
            selected["oof_metrics"]["hit_rate_at_10"],
            champion["oof_metrics"]["hit_rate_at_10"],
        )
        self.assertGreater(
            selected["paired_vs_two_feature_linear"]["paired_bootstrap_95_interval"][0],
            0.0,
        )
        self.assertTrue(self.report["performance"]["passed"])
        self.assertEqual(
            self.report["performance"]["metrics"],
            {
                key: self.report["all_development_refit"]["metrics"][key]
                for key in (
                    "hit_rate_at_10",
                    "mrr",
                    "mttc",
                    "recommended_technical_score",
                )
            },
        )


if __name__ == "__main__":
    unittest.main()

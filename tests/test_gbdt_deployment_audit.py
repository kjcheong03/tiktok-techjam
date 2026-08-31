from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/reports/gbdt_deployment_audit_v1.json"


class GBDTDeploymentAuditReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def test_oof_evidence_is_immutable_and_deployment_rule_is_frozen(self) -> None:
        self.assertEqual(
            self.report["immutable_oof_family_evidence"]["metrics"][
                "recommended_technical_score"
            ],
            0.861417,
        )
        self.assertEqual(self.report["deployable_round_rule"]["rounds"], 56)
        self.assertFalse(self.report["alternative_round_aggregations_evaluated"])
        self.assertFalse(self.report["holdout_accessed"])

    def test_all_executable_gates_pass(self) -> None:
        self.assertTrue(self.report["all_gates_passed"])
        self.assertEqual(self.report["decision"], "INTEGRATION_READY")
        for name in ("fold", "scenario", "determinism", "parity", "packaging"):
            self.assertTrue(self.report["gates"][name]["passed"], name)
        self.assertEqual(len(self.report["gates"]["fold"]["checks"]), 5)
        self.assertEqual(len(self.report["gates"]["scenario"]["checks"]), 4)

    def test_determinism_and_runtime_parity_are_exact(self) -> None:
        deterministic = self.report["gates"]["determinism"]
        self.assertTrue(deterministic["model_bytes_identical"])
        self.assertEqual(
            deterministic["first_model_sha256"],
            deterministic["second_model_sha256"],
        )
        self.assertEqual(deterministic["session_outcomes"]["mismatch_count"], 0)
        self.assertEqual(self.report["gates"]["parity"]["mismatches"], {})

    def test_response_failures_are_measured_not_assumed(self) -> None:
        deployment = self.report["deployable_refit"]["response_instrumentation"]
        runtime = self.report["isolated_runtime"]
        self.assertEqual(deployment["response_calls"], 353)
        self.assertEqual(runtime["response_calls"], 353)
        self.assertEqual(runtime["turn_count"], 353)
        self.assertEqual(deployment["failure_count"], 0)
        self.assertEqual(runtime["failure_count"], 0)
        self.assertEqual(sum(deployment["failure_counts"].values()), 0)
        self.assertEqual(sum(runtime["failure_counts"].values()), 0)

    def test_versioned_model_and_source_report_hashes_match(self) -> None:
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        deployable = self.report["deployable_refit"]
        self.assertEqual(
            digest(ROOT / deployable["model_path"]), deployable["model_sha256"]
        )
        self.assertEqual(
            digest(ROOT / self.report["source_oof_report"]),
            self.report["source_oof_report_sha256"],
        )
        for relative, expected in self.report["provenance"]["code_hashes"].items():
            self.assertEqual(digest(ROOT / relative), expected)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/gbdt_constraint_override_guard_v1.json"


class GBDTConstraintOverrideGuardReportTests(unittest.TestCase):
    report: Any

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_holdout_amendment_and_controls_are_frozen(self) -> None:
        self.assertFalse(self.report["holdout_accessed"])
        self.assertTrue(self.report["base"]["exactly_reproduced"])
        self.assertTrue(self.report["unguarded_v2"]["exactly_reproduced"])
        amendment = ROOT / self.report["amendment_path"]
        self.assertEqual(
            hashlib.sha256(amendment.read_bytes()).hexdigest(),
            self.report["amendment_sha256"],
        )

    def test_guard_passes_gates_but_preserves_uncertainty(self) -> None:
        promotion = self.report["promotion"]
        self.assertTrue(promotion["all_gates_passed"])
        self.assertEqual(promotion["decision"], "PROMOTE")
        self.assertTrue(all(promotion["gate_results"].values()))
        lower, upper = self.report["guarded_v2"]["paired_vs_base"][
            "paired_bootstrap_95_interval"
        ]
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)

    def test_routing_is_exact_and_observable(self) -> None:
        route = self.report["routing"]["aggregate"]
        self.assertEqual(route["fallback_sessions"], 22)
        self.assertEqual(route["fallback_turns"], 25)
        self.assertEqual(
            route["fallback_sessions_by_reason"],
            {"earlier_preference_override": 22},
        )
        self.assertEqual(
            route["fallback_sessions_by_scenario"], {"intent_override": 22}
        )
        self.assertEqual(
            self.report["guarded_v2"]["scenario_reward_deltas_vs_base"][
                "intent_override"
            ],
            0.0,
        )
        self.assertTrue(self.report["determinism"]["sessions_and_routing_exact"])

    def test_sessions_and_hashes_match(self) -> None:
        identifiers = [
            str(session["sample_id"])
            for session in self.report["guarded_v2"]["oof_sessions"]
        ]
        self.assertEqual(len(identifiers), 150)
        self.assertEqual(len(set(identifiers)), 150)

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        for relative, expected in self.report["code_hashes"].items():
            self.assertEqual(digest(ROOT / relative), expected)
        assets = self.report["model_assets"]
        self.assertEqual(digest(ROOT / assets["base_path"]), assets["base_sha256"])
        self.assertEqual(
            digest(ROOT / assets["constraint_path"]), assets["constraint_sha256"]
        )
        self.assertFalse(self.report["runtime"]["contention_affected"])


if __name__ == "__main__":
    unittest.main()

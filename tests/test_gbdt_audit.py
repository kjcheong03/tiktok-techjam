from __future__ import annotations

import json
import unittest
from pathlib import Path

from ghostlab.evaluation.gbdt_audit import (
    evidence_gates,
    metric_parity,
    packaging_gate,
    session_parity,
)
from scripts.measure_gbdt_runtime import TimedAgent

ROOT = Path(__file__).resolve().parents[1]


class _FailureAgent:
    def __init__(self, behavior: str) -> None:
        self.behavior = behavior

    def reset(self, session_id: str, user_profile: dict) -> None:
        return None

    def respond(
        self, session_id: str, user_message: str, turn: int, top_k: int
    ) -> object:
        if self.behavior == "raise":
            raise RuntimeError("instrument me")
        if self.behavior == "invalid":
            return None
        return {"message": "ok", "recommendations": []}


class GBDTAuditTests(unittest.TestCase):
    def test_amendment_freezes_median_without_alternative_search(self) -> None:
        amendment = json.loads(
            (ROOT / "configs/experiments/gbdt_reranker_v1_amendment_1.json").read_text()
        )
        rounds = amendment["outer_selected_rounds_in_frozen_fold_order"]
        self.assertEqual(rounds, [100, 4, 56, 98, 34])
        self.assertEqual(sorted(rounds)[len(rounds) // 2], 56)
        self.assertEqual(amendment["frozen_deployable_round_rule"]["rounds"], 56)
        self.assertFalse(
            amendment["frozen_deployable_round_rule"][
                "alternative_aggregations_may_be_compared_against_all_development_outcomes"
            ]
        )

    def test_existing_oof_fold_and_scenario_gates_execute(self) -> None:
        report = json.loads(
            (ROOT / "artifacts/reports/gbdt_reranker_v1.json").read_text()
        )
        gates = evidence_gates(
            report,
            candidate_id="shallow_metadata_depth3",
            scenario_delta_floor=-0.005,
        )
        self.assertTrue(gates["fold"]["passed"])
        self.assertTrue(gates["scenario"]["passed"])
        self.assertEqual(len(gates["fold"]["checks"]), 5)
        self.assertEqual(len(gates["scenario"]["checks"]), 4)

    def test_failure_instrumentation_counts_exceptions_and_invalid_outputs(
        self,
    ) -> None:
        raising = TimedAgent(_FailureAgent("raise"))  # type: ignore[arg-type]
        response = raising.respond("s", "q", 1, 10)
        self.assertEqual(response["message"], "")
        self.assertEqual(raising.response_exception_count, 1)
        self.assertEqual(raising.failure_count, 1)

        invalid = TimedAgent(_FailureAgent("invalid"))  # type: ignore[arg-type]
        response = invalid.respond("s", "q", 1, 10)
        self.assertEqual(response["message"], "")
        self.assertEqual(invalid.invalid_response_count, 1)
        self.assertEqual(invalid.failure_count, 1)

    def test_packaging_and_parity_gates_fail_closed(self) -> None:
        limits = {
            "cold_start_seconds_max": 30.0,
            "warm_turn_p95_ms_max": 500.0,
            "peak_process_memory_mb_max": 4096.0,
            "model_asset_mb_max": 500.0,
            "external_calls_per_turn_max": 0,
            "response_failure_count_max": 0,
        }
        measurement = {
            "cold_start_seconds": 1.0,
            "warm_turn_p95_ms": 2.0,
            "peak_process_memory_mb": 3.0,
            "model_asset_mb": 0.1,
            "external_calls_per_turn": 0,
            "failure_count": 1,
        }
        gate = packaging_gate(measurement, limits)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["response_failures"])

        metrics = {
            "hit_rate_at_10": 1.0,
            "mrr": 0.5,
            "mttc": 2.0,
            "recommended_technical_score": 0.8,
        }
        self.assertTrue(metric_parity(metrics, dict(metrics))["passed"])
        changed = {**metrics, "mrr": 0.4}
        self.assertFalse(metric_parity(metrics, changed)["passed"])

    def test_session_parity_reports_exact_mismatch(self) -> None:
        first = [{"sample_id": "one", "hit": True}]
        second = [{"sample_id": "one", "hit": False}]
        result = session_parity(first, second)
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()

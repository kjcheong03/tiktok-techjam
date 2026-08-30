from __future__ import annotations

import pytest

from scripts.evaluate_adaptive_holdout import (
    build_fair_holdout_report,
    compare_reports,
)


def _report(*, score: float, buying: float, violations: int = 0) -> dict:
    return {
        "recommended_technical_score": score,
        "hit_rate_at_10": score,
        "mrr": score,
        "mttc": 2.0,
        "scenario_metrics": {
            "buying": {"hit_rate_at_10": buying, "mrr": buying},
            "browsing": {"hit_rate_at_10": score, "mrr": score},
        },
        "source_metrics": {
            "data/public_set.jsonl": {"hit_rate_at_10": score, "mrr": score}
        },
        "route_metrics": {
            "buying": {"hit_rate_at_10": buying, "mrr": buying},
            "browsing": {"hit_rate_at_10": score, "mrr": score},
        },
        "adaptive_runtime": {
            "trace_count": 10,
            "fallback_count": 0,
            "output_constraint_violation_count": violations,
            "overload_cutoff_trace_violations": 0,
        },
        "target_survival_audit": {"confirmed_target_removal_count": 0},
    }


def _gates() -> dict:
    return {
        "combined_score_min_delta": 0.0,
        "buying_hit_at_10_max_regression": 0.01,
        "buying_mrr_max_regression": 0.01,
        "scenario_hit_at_10_max_regression": 0.02,
        "route_hit_at_10_max_regression": 0.02,
        "source_hit_at_10_max_regression": 0.02,
        "mttc_max_regression": 0.5,
        "fallback_rate_max_regression": 0.01,
        "require_zero_output_constraint_violations": True,
        "require_zero_confirmed_target_removals": True,
        "require_zero_overload_trace_violations": True,
    }


def test_holdout_gates_accept_like_for_like_improvement() -> None:
    rows = compare_reports(
        _report(score=0.82, buying=0.90),
        _report(score=0.80, buying=0.90),
        _gates(),
    )
    assert rows
    assert all(row["passed"] for row in rows)


def test_holdout_gates_reject_constraint_violation() -> None:
    rows = compare_reports(
        _report(score=0.82, buying=0.90, violations=1),
        _report(score=0.80, buying=0.90),
        _gates(),
    )
    failed = {row["gate"] for row in rows if not row["passed"]}
    assert "zero_output_constraint_violations" in failed


def _fair_report(score: float) -> dict:
    return {
        "evaluation_contract": {
            "harness_id": "shared-v1",
            "contract_sha256": "same-contract",
        },
        "hit_rate_at_10": score,
        "mrr": score,
        "mttc": 2.0,
        "recommended_technical_score": score,
        "scenario_metrics": {"buying": {"hit_rate_at_10": score}},
        "source_metrics": {"public": {"hit_rate_at_10": score}},
        "sessions": [
            {
                "sample_id": f"holdout_{index:04d}",
                "scenario_type": "buying",
                "hit": True,
                "first_hit_turn": 1,
                "best_rank": 1,
                "reciprocal_rank": 1.0,
            }
            for index in range(550)
        ],
    }


def test_fair_holdout_report_keeps_references_out_of_promotion() -> None:
    reports = [_fair_report(score) for score in (0.4, 0.6, 0.8, 0.82)]
    report = build_fair_holdout_report(
        frozen={
            "candidate_id": "challenger-test",
            "config_path": "configs/finalist.json",
            "config_sha256": "file-hash",
        },
        reference_a=reports[0],
        reference_b=reports[1],
        control=reports[2],
        challenger=reports[3],
        reference_a_path="a.json",
        reference_b_path="b.json",
        control_path="c.json",
        challenger_path="d.json",
        control_config="configs/control.json",
        control_config_sha256="control-hash",
        challenger_config_canonical_sha256="challenger-canonical-hash",
        gate_results=[{"gate": "score", "passed": True}],
        paired={"mean_paired_delta": 0.02},
        pairwise={
            "B_minus_A": {"mean_paired_delta": 0.2},
            "C_minus_B": {"mean_paired_delta": 0.2},
            "D_minus_C": {"mean_paired_delta": 0.02},
        },
        receipt_path="receipt.json",
    )

    assert report["system_count"] == 4
    assert report["reference_count"] == 2
    assert report["comparison_semantics"]["same_ground"] is True
    assert [item["champion_eligible"] for item in report["systems"]] == [
        False,
        False,
        True,
        True,
    ]
    assert report["promotion_comparison"]["control_system_id"].startswith("C_")
    assert report["challenger"]["config_sha256"] == "challenger-canonical-hash"
    assert report["challenger"]["config_file_sha256"] == "file-hash"
    assert report["decision"] == "PROMOTE"


def test_fair_holdout_report_rejects_mismatched_session_order() -> None:
    reports = [_fair_report(score) for score in (0.4, 0.6, 0.8, 0.82)]
    reports[1]["sessions"] = list(reversed(reports[1]["sessions"]))
    with pytest.raises(ValueError, match="same 550 ordered session IDs"):
        build_fair_holdout_report(
            frozen={
                "candidate_id": "challenger-test",
                "config_path": "configs/finalist.json",
                "config_sha256": "file-hash",
            },
            reference_a=reports[0],
            reference_b=reports[1],
            control=reports[2],
            challenger=reports[3],
            reference_a_path="a.json",
            reference_b_path="b.json",
            control_path="c.json",
            challenger_path="d.json",
            control_config="configs/control.json",
            control_config_sha256="control-hash",
            challenger_config_canonical_sha256="challenger-hash",
            gate_results=[],
            paired={},
            pairwise={},
            receipt_path="receipt.json",
        )


def test_fair_holdout_report_rejects_different_evaluator_contract() -> None:
    reports = [_fair_report(score) for score in (0.4, 0.6, 0.8, 0.82)]
    reports[3]["evaluation_contract"] = {
        "harness_id": "different-v2",
        "contract_sha256": "different-contract",
    }
    with pytest.raises(ValueError, match="identical evaluation contract"):
        build_fair_holdout_report(
            frozen={
                "candidate_id": "challenger-test",
                "config_path": "configs/finalist.json",
                "config_sha256": "file-hash",
            },
            reference_a=reports[0],
            reference_b=reports[1],
            control=reports[2],
            challenger=reports[3],
            reference_a_path="a.json",
            reference_b_path="b.json",
            control_path="c.json",
            challenger_path="d.json",
            control_config="configs/control.json",
            control_config_sha256="control-hash",
            challenger_config_canonical_sha256="challenger-hash",
            gate_results=[],
            paired={},
            pairwise={},
            receipt_path="receipt.json",
        )

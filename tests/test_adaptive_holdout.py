from __future__ import annotations

from scripts.evaluate_adaptive_holdout import compare_reports


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

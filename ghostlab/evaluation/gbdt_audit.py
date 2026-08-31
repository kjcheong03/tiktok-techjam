from __future__ import annotations

import statistics
from collections import defaultdict

from evaluator.local_evaluator import metric_summary
from ghostlab.research.replay import session_reward


def _technical_score(sessions: list[dict]) -> float:
    return round(statistics.fmean(session_reward(item) for item in sessions), 6)


def evidence_gates(
    report: dict,
    *,
    candidate_id: str,
    scenario_delta_floor: float,
) -> dict[str, dict[str, object]]:
    candidate = report["variants"][candidate_id]
    control = report["controls"]["two_feature_linear_champion"]
    control_folds = {int(fold["outer_fold"]): fold for fold in control["folds"]}
    fold_checks = []
    for fold in candidate["folds"]:
        outer_fold = int(fold["outer_fold"])
        candidate_score = float(fold["outer_metrics"]["recommended_technical_score"])
        control_score = float(
            control_folds[outer_fold]["outer_metrics"]["recommended_technical_score"]
        )
        fold_checks.append(
            {
                "outer_fold": outer_fold,
                "candidate_score": candidate_score,
                "control_score": control_score,
                "delta": round(candidate_score - control_score, 6),
                "passed": candidate_score > control_score,
            }
        )

    def by_scenario(sessions: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for session in sessions:
            grouped[str(session["scenario_type"])].append(session)
        return dict(grouped)

    candidate_scenarios = by_scenario(candidate["oof_sessions"])
    control_scenarios = by_scenario(control["oof_sessions"])
    scenario_checks = []
    for scenario in sorted(candidate_scenarios):
        candidate_sessions = candidate_scenarios[scenario]
        control_sessions = control_scenarios[scenario]
        candidate_metrics = metric_summary(candidate_sessions)
        control_metrics = metric_summary(control_sessions)
        candidate_technical = _technical_score(candidate_sessions)
        control_technical = _technical_score(control_sessions)
        hit_passed = float(candidate_metrics["hit_rate_at_10"]) >= float(
            control_metrics["hit_rate_at_10"]
        )
        technical_delta = round(candidate_technical - control_technical, 6)
        scenario_checks.append(
            {
                "scenario": scenario,
                "candidate_hit_rate_at_10": candidate_metrics["hit_rate_at_10"],
                "control_hit_rate_at_10": control_metrics["hit_rate_at_10"],
                "candidate_technical_score": candidate_technical,
                "control_technical_score": control_technical,
                "technical_score_delta": technical_delta,
                "hit_rate_passed": hit_passed,
                "technical_score_passed": technical_delta >= scenario_delta_floor,
                "passed": hit_passed and technical_delta >= scenario_delta_floor,
            }
        )
    return {
        "fold": {
            "rule": "candidate technical score exceeds matched linear control in every outer fold",
            "checks": fold_checks,
            "passed": all(item["passed"] for item in fold_checks),
        },
        "scenario": {
            "rule": "no scenario Hit@10 regression and technical-score delta >= floor",
            "technical_score_delta_floor": scenario_delta_floor,
            "checks": scenario_checks,
            "passed": all(item["passed"] for item in scenario_checks),
        },
    }


def packaging_gate(measurement: dict, limits: dict) -> dict[str, object]:
    checks = {
        "cold_start": float(measurement["cold_start_seconds"])
        <= float(limits["cold_start_seconds_max"]),
        "warm_turn_p95": float(measurement["warm_turn_p95_ms"])
        <= float(limits["warm_turn_p95_ms_max"]),
        "peak_memory": float(measurement["peak_process_memory_mb"])
        <= float(limits["peak_process_memory_mb_max"]),
        "model_asset": float(measurement["model_asset_mb"])
        <= float(limits["model_asset_mb_max"]),
        "external_calls": int(measurement["external_calls_per_turn"])
        <= int(limits["external_calls_per_turn_max"]),
        "response_failures": int(measurement["failure_count"])
        <= int(limits["response_failure_count_max"]),
    }
    return {"checks": checks, "passed": all(checks.values())}


def metric_parity(left: dict, right: dict) -> dict[str, object]:
    keys = (
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "recommended_technical_score",
    )
    mismatches = {
        key: {"research": left[key], "isolated_runtime": right[key]}
        for key in keys
        if left[key] != right[key]
    }
    return {"mismatches": mismatches, "passed": not mismatches}


def session_parity(left: list[dict], right: list[dict]) -> dict[str, object]:
    left_by_id = {str(item["sample_id"]): item for item in left}
    right_by_id = {str(item["sample_id"]): item for item in right}
    mismatches = {
        sample_id: {
            "first": left_by_id.get(sample_id),
            "second": right_by_id.get(sample_id),
        }
        for sample_id in sorted(set(left_by_id) | set(right_by_id))
        if left_by_id.get(sample_id) != right_by_id.get(sample_id)
    }
    return {
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }

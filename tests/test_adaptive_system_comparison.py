from __future__ import annotations

from scripts.build_adaptive_system_comparison import build_comparison


def _report(score: float, rank: int) -> dict:
    sessions = [
        {
            "sample_id": f"sample_{index:04d}",
            "scenario_type": "buying",
            "hit": True,
            "first_hit_turn": 1,
            "best_rank": rank,
            "reciprocal_rank": 1.0 / rank,
        }
        for index in range(1650)
    ]
    return {
        "evaluation_contract": {
            "harness_id": "shared-v1",
            "contract_sha256": "same-contract",
        },
        "metrics": {
            "hit_rate_at_10": 1.0,
            "mrr": 1.0 / rank,
            "mttc": 1.0,
            "recommended_technical_score": score,
        },
        "sessions": sessions,
    }


def test_comparison_keeps_baselines_out_of_champion_selection() -> None:
    top_three = {
        "finalists": [
            {
                "rank": 1,
                "candidate_id": "challenger-test",
                "config_path": "configs/finalist.json",
                "config_sha256": "config-hash",
                "promotion_eligible": True,
                "techniques": ["state.baseline_v2"],
                "metrics": {
                    "score": 0.9,
                    "hit_rate_at_10": 1.0,
                    "mrr": 0.8,
                    "mttc": 1.0,
                    "mean_paired_delta": 0.02,
                },
            }
        ]
    }
    report_a = _report(0.6, 3)
    report_b = _report(0.7, 2)
    report_c = _report(0.8, 1)
    finalist_evaluations = {
        "evaluations": [
            {
                "candidate_id": "challenger-test",
                "config_path": "configs/finalist.json",
                "config_sha256": "config-hash",
                "report": _report(0.85, 1),
            }
        ]
    }
    report = build_comparison(
        report_a,
        report_b,
        report_c,
        top_three,
        finalist_evaluations,
    )

    assert report["sample_count"] == 1650
    assert report["holdout_accessed"] is False
    assert (
        report["comparison_semantics"]["champion_selection_scope"] == "C versus D only"
    )
    assert [item["champion_eligible"] for item in report["systems"][:3]] == [
        False,
        False,
        True,
    ]
    assert report["systems"][3]["system_id"] == "D1_challenger-test"
    assert report["systems"][3]["sessions"]
    assert report["systems"][3]["ghostlab_selection_metrics"]["score"] == 0.9
    assert report["ghostlab_status"] == "top_three_available"

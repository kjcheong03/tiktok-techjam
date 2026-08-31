from __future__ import annotations

import pytest

from scripts.show_optimizer_summary import build_summary


def _fixtures() -> tuple[dict, dict]:
    campaign = {
        "mode": "race",
        "search_mode": "additive_warm_start",
        "sample_count": 1650,
        "stage_counts": {"f0": 14, "f1": 5, "f2": 3},
        "fidelity_sample_counts": {"f0": 330, "f1": 825, "f2": 1650},
        "elapsed_seconds": 16_706.0,
    }
    finalists = {
        "campaign_report": (
            "artifacts/reports/adaptive_hybrid_additive_warm_start_1650.json"
        ),
        "finalists": [
            {
                "candidate_id": "candidate-1",
                "config_sha256": "a" * 64,
                "promotion_eligible": True,
                "constraint_violations": 0,
                "techniques": [
                    "fusion.rrf",
                    "ranking.local_llm_semantic.v1",
                    "ranking.top10_residual_reranker.v2",
                ],
                "metrics": {
                    "hit_rate_at_10": 0.96,
                    "mrr": 0.70,
                    "mttc": 2.75,
                    "score": 0.86,
                },
            }
        ],
    }
    return campaign, finalists


def test_build_summary_prints_progressive_race_and_winner() -> None:
    campaign, finalists = _fixtures()

    summary = build_summary(campaign, finalists)

    assert "1,650 (final-selection set not accessed)" in summary
    assert "F0: 14 candidates × 330 sessions" in summary
    assert "F1:  5 candidates × 825 sessions" in summary
    assert "F2:  3 candidates × 1,650 sessions" in summary
    assert "RRF evidence fusion" in summary
    assert "Top-10 residual reranker" in summary
    assert "Promotion eligible: yes" in summary


def test_build_summary_rejects_non_race_report() -> None:
    campaign, finalists = _fixtures()
    campaign["mode"] = "plan_only"

    with pytest.raises(ValueError, match="completed GhostLab race"):
        build_summary(campaign, finalists)

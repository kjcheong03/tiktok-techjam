from __future__ import annotations

import json

import numpy as np

from scripts.run_membership_preserving_residual import (
    RESIDUAL_FEATURES,
    ModelSpec,
    TraceSession,
    TraceTurn,
)
from scripts.run_state_v2_adaptive_residual import (
    ROOT,
    _rank_aware_dataset,
    _robust_evidence,
)


def _turn(labels: list[int], turn: int) -> TraceTurn:
    count = len(labels)
    return TraceTurn(
        sample_id=f"sample-{turn}",
        turn=turn,
        eligible=True,
        target="target",
        ranking=tuple(f"item-{index}" for index in range(count)),
        features=np.zeros((count, len(RESIDUAL_FEATURES)), dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
    )


def test_rank_aware_weights_prioritize_late_targets_and_downweight_misses() -> None:
    sessions = [
        TraceSession(
            sample_id="session",
            scenario_type="buying",
            turns=(_turn([0, 0, 0, 1], 1), _turn([0, 0], 2)),
            parent_outcome={},
        )
    ]
    _, labels, weights = _rank_aware_dataset(
        ModelSpec("regularized_logistic", "rank", 0.2), sessions
    )

    assert labels.tolist() == [0, 0, 0, 1, 0, 0]
    assert weights.tolist() == [1.0, 1.0, 1.0, 4.0, 0.35, 0.35]


def test_robust_evidence_requires_stable_inner_fold_improvement() -> None:
    stable = _robust_evidence(
        {
            "score_delta": 0.02,
            "fold_deltas": [0.01, 0.02, 0.03, 0.01],
            "nonnegative_folds": 4,
            "worst_scenario_delta": 0.005,
            "behavior": {"activations": 10},
        },
        total_turns=100,
        inner_fold_count=4,
    )
    unstable = _robust_evidence(
        {
            "score_delta": 0.02,
            "fold_deltas": [0.08, -0.04, -0.02, 0.06],
            "nonnegative_folds": 2,
            "worst_scenario_delta": -0.01,
            "behavior": {"activations": 10},
        },
        total_turns=100,
        inner_fold_count=4,
    )

    assert stable["eligible"] is True
    assert float(stable["robust_utility"]) > 0.0
    assert unstable["eligible"] is False
    assert float(unstable["robust_utility"]) < float(stable["robust_utility"])


def test_definitive_report_is_fold_safe_and_membership_preserving() -> None:
    report_path = ROOT / "artifacts/reports/state_v2_adaptive_residual_v2.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["holdout_accessed"] is False
    assert report["outer_folds_completed"] == 5
    for parent_name in ("primary", "secondary"):
        parent = report[parent_name]
        assert parent["decision"]["promotion_gates_passed"] is True
        assert parent["decision"]["membership_failures"] == 0
        assert parent["decision"]["nonnegative_outer_folds"] == 5
        assert (
            parent["control"]["oof_metrics"]["hit_rate_at_10"]
            == parent["candidate"]["oof_metrics"]["hit_rate_at_10"]
        )
        assert (
            parent["control"]["oof_metrics"]["mttc"]
            == parent["candidate"]["oof_metrics"]["mttc"]
        )
        assert (
            parent["candidate"]["paired_vs_control"]["paired_bootstrap_95_interval"][0]
            > 0.0
        )
        assert len(parent["folds"]) == 5
        assert all(fold["fit_receipt"]["disjoint"] for fold in parent["folds"])

from __future__ import annotations

import pytest

from ghostlab.optimization.adaptive_hybrid import (
    AdaptiveArchitectureAudit,
    AdaptiveHybridBinding,
    AdaptiveHybridTrial,
    generate_adaptive_trials,
)
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig


def test_ghostlab_trial_changes_values_without_changing_architecture() -> None:
    baseline = AdaptiveHybridConfig()
    candidate = AdaptiveHybridBinding.materialize(
        baseline,
        AdaptiveHybridTrial(
            buying_retrieval_k=300,
            dense_output_k=250,
            semantic_weight=0.2,
            broad_discovery_turns=1,
        ),
        policy_id="trial_1",
    )
    assert candidate.policy_id == "trial_1"
    assert candidate.buying.retrieval_k == 300
    assert candidate.browsing.output_k == 250
    assert candidate.semantic_ranker.weight == 0.2
    assert candidate.state.component == baseline.state.component
    assert candidate.router.component == baseline.router.component
    assert candidate.orchestration == baseline.orchestration
    assert AdaptiveArchitectureAudit.validate(candidate) is candidate


def test_ghostlab_trial_cannot_disable_or_replace_required_slots() -> None:
    with pytest.raises(ValueError):
        AdaptiveHybridBinding.materialize(
            AdaptiveHybridConfig(),
            {"semantic_weight": 0.0},
            policy_id="invalid",
        )


def test_search_space_changes_values_but_preserves_every_required_slot() -> None:
    baseline = AdaptiveHybridConfig()
    trials = generate_adaptive_trials(baseline, 4)
    candidates = [
        AdaptiveHybridBinding.materialize(
            baseline, trial, policy_id=f"trial_{index}"
        )
        for index, trial in enumerate(trials)
    ]
    assert len({candidate.canonical_hash() for candidate in candidates}) == 4
    assert all(
        AdaptiveArchitectureAudit.validate(candidate) is candidate
        for candidate in candidates
    )
    assert all(candidate.orchestration == baseline.orchestration for candidate in candidates)
    with pytest.raises(ValueError):
        AdaptiveHybridBinding.materialize(
            AdaptiveHybridConfig(),
            {"semantic_backend": "off"},
            policy_id="invalid",
        )
    with pytest.raises(ValueError):
        AdaptiveHybridConfig.model_validate(
            {"semantic_ranker": {"activate_for_browsing": False}}
        )

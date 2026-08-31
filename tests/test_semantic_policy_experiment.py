from __future__ import annotations

from ghostlab.policy.signals import RetrievalSignals
from ghostlab.runtime.adaptive_components import (
    RouteDecision,
    SemanticActivationPolicy,
)
from ghostlab.runtime.adaptive_config import AdaptiveHybridConfig
from ghostlab.state.v2_view import ConstraintView, V2StateView
from scripts.run_semantic_activation_study import StudyActivationPolicy
from scripts.run_semantic_policy_experiment import _trial_matrix


def _view(*constraints: ConstraintView) -> V2StateView:
    return V2StateView(
        query_text="summer wedding clothing",
        active_constraints=tuple(constraints),
        intent_epoch=0,
        shown_ids=frozenset(),
        asked_attributes=(),
        no_preference_attributes=frozenset(),
        turn=1,
    )


def _constraint(attribute: str) -> ConstraintView:
    return ConstraintView(
        attribute=attribute,
        values=("breathable",),
        relation="equals",
        polarity="include",
        strength="hard",
        operator="equals",
        source_turn=1,
        provenance="explicit",
    )


def _policy(mode: str, *, maximum_margin: float = 0.02) -> StudyActivationPolicy:
    selected = SemanticActivationPolicy(AdaptiveHybridConfig().semantic_ranker)
    return StudyActivationPolicy(
        mode,
        selected,
        maximum_margin=maximum_margin,
        minimum_entropy=0.85,
    )


def test_ambiguity_gate_uses_margin_and_entropy() -> None:
    browsing = RouteDecision("browsing", 0.7, "test")
    ambiguous = RetrievalSignals(10, 0.01, 0.95, 0.2)
    confident = RetrievalSignals(10, 0.10, 0.95, 0.2)
    concentrated = RetrievalSignals(10, 0.01, 0.40, 0.2)

    assert _policy("browsing_ambiguous").decide(
        browsing, _view(), overloaded=False, signals=ambiguous
    ).active
    assert not _policy("browsing_ambiguous").decide(
        browsing, _view(), overloaded=False, signals=confident
    ).active
    assert not _policy("browsing_ambiguous").decide(
        browsing, _view(), overloaded=False, signals=concentrated
    ).active


def test_buying_gates_remain_route_and_constraint_safe() -> None:
    buying = RouteDecision("buying", 0.9, "test")
    browsing = RouteDecision("browsing", 0.7, "test")

    assert _policy("buying_all").decide(
        buying, _view(), overloaded=False
    ).active
    assert not _policy("buying_all").decide(
        browsing, _view(), overloaded=False
    ).active
    assert not _policy("buying_all").decide(
        buying, _view(), overloaded=True
    ).active
    assert _policy("buying_semantic_constraints").decide(
        buying, _view(_constraint("feature")), overloaded=False
    ).active
    assert not _policy("buying_semantic_constraints").decide(
        buying, _view(_constraint("color")), overloaded=False
    ).active


def test_experiment_matrix_is_bounded_and_contains_controls() -> None:
    trials = _trial_matrix()
    assert len(trials) == 12
    assert {item["trial_id"] for item in trials} >= {
        "browsing_control_no_llm",
        "buying_control_no_llm",
        "browsing_gated_margin_0.02_weight_0.05",
        "buying_all_weight_0.05",
    }
    assert all(float(item["weight"]) <= 0.15 for item in trials)

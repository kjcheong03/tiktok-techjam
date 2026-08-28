from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ghostlab.state.baseline_v2 import StateBaselineV2, StructuredConstraint
from ghostlab.state.v2_view import V2SessionController


def constraint(attribute: str, value: str, turn: int) -> StructuredConstraint:
    return StructuredConstraint(
        attribute=attribute,  # type: ignore[arg-type]
        values=[value],
        source_turn=turn,
        source_text=value,
    )


def test_snapshot_is_immutable_and_contains_only_copied_observable_state() -> None:
    state = StateBaselineV2("session", {})
    state.apply_constraints(
        [constraint("category", "shoes", 1), constraint("color", "navy", 1)]
    )
    state.asked_attributes.append("size")
    controller = V2SessionController(state)
    controller.record_shown(["A", "B"])

    view = controller.snapshot(query_text=state.build_state_query(), turn=2)

    assert view.query_text == "shoes. navy"
    assert view.shown_ids == frozenset({"A", "B"})
    assert view.asked_attributes == ("size",)
    assert view.positive_constraints() == {
        "category": ["shoes"],
        "color": ["navy"],
    }
    state.active_constraints[1].values.append("red")
    assert view.active_constraints[1].values == ("navy",)
    with pytest.raises(FrozenInstanceError):
        view.turn = 3  # type: ignore[misc]


def test_controller_owns_history_and_resets_it_on_intent_epoch_change() -> None:
    state = StateBaselineV2("session", {})
    state.observe(
        "black shoes",
        1,
        parsed_constraints=[constraint("color", "black", 1)],
    )
    controller = V2SessionController(state)
    controller.record_shown(["A", "B"])
    assert controller.filter_ranking(["A", "C", "B", "D"]) == ["C", "D"]

    state.observe(
        "Actually, navy instead.",
        2,
        parsed_constraints=[constraint("color", "navy", 2)],
    )
    view = controller.snapshot(query_text=state.build_state_query(), turn=2)

    assert state.intent_epoch == 1
    assert view.shown_ids == frozenset()
    assert controller.filter_ranking(["A", "B"]) == ["A", "B"]

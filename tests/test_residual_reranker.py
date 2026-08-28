from __future__ import annotations

import pytest

from ghostlab.retrieval.residual import (
    ResidualPolicy,
    membership_preserving_reorder,
)


def test_reorders_without_changing_membership() -> None:
    original = ["A", "B", "C", "D"]
    result = membership_preserving_reorder(
        original,
        [0.1, 0.8, 0.2, 0.05],
        ResidualPolicy(rerank_depth=3, maximum_moved_ids=3),
    )
    assert result.activated
    assert result.ranking == ("B", "C", "A", "D")
    assert set(result.ranking) == set(original)


def test_confidence_gate_preserves_original_exactly() -> None:
    original = ["A", "B", "C"]
    result = membership_preserving_reorder(
        original,
        [0.4, 0.41, 0.39],
        ResidualPolicy(minimum_probability_margin=0.05),
    )
    assert not result.activated
    assert result.reason == "probability_margin"
    assert result.ranking == tuple(original)


def test_movement_gate_preserves_original_exactly() -> None:
    original = ["A", "B", "C", "D"]
    result = membership_preserving_reorder(
        original,
        [0.1, 0.2, 0.3, 0.9],
        ResidualPolicy(maximum_moved_ids=2),
    )
    assert not result.activated
    assert result.reason == "movement_limit"
    assert result.ranking == tuple(original)


def test_duplicate_membership_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        membership_preserving_reorder(["A", "A"], [0.1, 0.2], ResidualPolicy())


def test_target_presence_is_invariant_for_every_possible_target() -> None:
    original = ["A", "B", "C", "D"]
    result = membership_preserving_reorder(
        original,
        [0.1, 0.8, 0.2, 0.05],
        ResidualPolicy(rerank_depth=4, maximum_moved_ids=4),
    )
    for target in [*original, "ABSENT"]:
        assert (target in result.ranking) == (target in original)

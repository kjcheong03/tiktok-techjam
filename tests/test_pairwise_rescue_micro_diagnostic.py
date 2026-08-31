from __future__ import annotations

from dataclasses import dataclass

from scripts.run_pairwise_rescue_micro_diagnostic import (
    EXPECTED_OPPORTUNITIES,
    BatchPairwiseScorer,
    Comparison,
    PairwiseBoundaryPromoter,
    TurnPoint,
    matched_negative_eligible,
    select_micro_cases,
)


@dataclass(frozen=True)
class AuthorityResult:
    ranking: tuple[str, ...]
    violation_count: int = 0


class AllowAllAuthority:
    def enforce(self, ranking, view):
        del view
        return AuthorityResult(tuple(ranking))


class RejectOutsiderAuthority:
    def enforce(self, ranking, view):
        del view
        kept = tuple(item for item in ranking if item != "P11")
        return AuthorityResult(kept, len(ranking) - len(kept))


class ScriptedScorer(BatchPairwiseScorer):
    model_id = "scripted"

    def __init__(self, winners: set[str]) -> None:
        self.winners = winners

    def compare(self, query, ordered_pairs):
        del query
        return tuple(
            Comparison(
                left,
                right,
                next((item for item in (left, right) if item in self.winners), None),
                1.0,
            )
            for left, right in ordered_pairs
        )


def ranking():
    return tuple(f"P{rank}" for rank in range(1, 15))


def test_protocol_has_exactly_twelve_order_balanced_comparisons() -> None:
    pairs = PairwiseBoundaryPromoter.ordered_pairs(ranking())
    assert len(pairs) == 12
    for outsider in ("P11", "P12", "P13"):
        for incumbent in ("P9", "P10"):
            assert (outsider, incumbent) in pairs
            assert (incumbent, outsider) in pairs


def test_outsider_must_win_against_both_incumbents_in_both_orders() -> None:
    promoter = PairwiseBoundaryPromoter(
        ScriptedScorer({"P11"}),
        AllowAllAuthority(),  # type: ignore[arg-type]
    )
    result = promoter.promote("query", ranking(), object())  # type: ignore[arg-type]
    assert result.promoted_id == "P11"
    assert result.ranking[:8] == ranking()[:8]
    assert result.ranking[8] == "P9"
    assert result.ranking[9] == "P11"
    assert len(result.ranking) == len(ranking())
    assert set(result.ranking) == set(ranking())


def test_partial_pairwise_wins_do_not_promote() -> None:
    class PartialScorer(BatchPairwiseScorer):
        model_id = "partial"

        def compare(self, query, ordered_pairs):
            del query
            return tuple(
                Comparison(
                    left,
                    right,
                    "P11" if "P11" in (left, right) and "P9" in (left, right) else None,
                    1.0,
                )
                for left, right in ordered_pairs
            )

    result = PairwiseBoundaryPromoter(
        PartialScorer(),
        AllowAllAuthority(),  # type: ignore[arg-type]
    ).promote("query", ranking(), object())  # type: ignore[arg-type]
    assert result.promoted_id is None
    assert result.ranking == ranking()


def test_constraint_revalidation_blocks_promotion() -> None:
    result = PairwiseBoundaryPromoter(
        ScriptedScorer({"P11"}),
        RejectOutsiderAuthority(),  # type: ignore[arg-type]
    ).promote("query", ranking(), object())  # type: ignore[arg-type]
    assert result.promoted_id is None
    assert result.ranking == ranking()
    assert result.constraint_violation_count == 1


def test_scorer_failure_falls_back_to_exact_noop() -> None:
    class BrokenScorer(BatchPairwiseScorer):
        model_id = "broken"

        def compare(self, query, ordered_pairs):
            raise RuntimeError("expected failure")

    result = PairwiseBoundaryPromoter(
        BrokenScorer(),
        AllowAllAuthority(),  # type: ignore[arg-type]
    ).promote("query", ranking(), object())  # type: ignore[arg-type]
    assert result.ranking == ranking()
    assert result.promoted_id is None
    assert result.failure_reason == "RuntimeError: expected failure"


def _point(
    sample_id: str, target: str, rank: int, pool_size: int, *, turn: int = 2
) -> TurnPoint:
    values = [f"{sample_id}-P{index}" for index in range(1, pool_size + 1)]
    values[rank - 1] = target
    return TurnPoint(
        sample_id,
        f"runtime-{sample_id}",
        turn,
        "query",
        "browsing",
        tuple(values),
        target,
        object(),  # type: ignore[arg-type]
    )


def test_micro_set_has_four_rank_1_to_8_matched_negatives() -> None:
    opportunities = [
        _point(sample_id, target, rank, pool_size)
        for (sample_id, _), (target, rank, pool_size) in EXPECTED_OPPORTUNITIES.items()
    ]
    negatives = [
        _point(f"negative-{index}", f"TARGET-{index}", index, 275 + index * 10)
        for index in range(1, 5)
    ]
    ineligible = [
        _point("rank-nine", "TARGET-9", 9, 270),
        _point("wrong-turn", "TARGET-W", 1, 270, turn=3),
    ]

    cases, evidence = select_micro_cases([*opportunities, *negatives, *ineligible])

    assert evidence["opportunity_count"] == 2
    assert evidence["matched_negative_count"] == 4
    selected_negatives = cases[2:]
    assert len(selected_negatives) == 4
    assert all(
        item.target_rank is not None and item.target_rank <= 8
        for item in selected_negatives
    )
    assert not matched_negative_eligible(ineligible[0], opportunities[0], set())
    assert not matched_negative_eligible(ineligible[1], opportunities[0], set())

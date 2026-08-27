from __future__ import annotations

import json
from pathlib import Path

from ghostlab.retrieval.diversify import (
    DiversificationContext,
    FacetMMRDiversifier,
)


def write_catalog(path: Path) -> None:
    products = [
        {"parent_asin": "a", "categories": ["shoe"], "details": {"color": "red"}},
        {"parent_asin": "b", "categories": ["shoe"], "details": {"color": "red"}},
        {"parent_asin": "c", "categories": ["boot"], "details": {"color": "blue"}},
        {"parent_asin": "d", "categories": ["sandal"], "details": {"color": "green"}},
    ]
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products), encoding="utf-8"
    )


def test_early_mmr_preserves_first_and_candidate_set(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    diversifier = FacetMMRDiversifier(
        catalog, relevance_weight=0.5, rerank_k=4, output_k=3
    )
    original = ["a", "b", "c", "d"]
    decision = diversifier.rerank(
        original, DiversificationContext(turn=1, active_constraint_count=0)
    )
    assert decision.activated
    assert decision.ranking[0] == "a"
    assert set(decision.ranking) == set(original)
    assert decision.ranking.index("c") < decision.ranking.index("b")


def test_diversification_turns_off_for_specific_or_late_intent(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    write_catalog(catalog)
    diversifier = FacetMMRDiversifier(catalog, max_turn=2, max_active_constraints=1)
    ranking = ["a", "b", "c"]
    for context in (
        DiversificationContext(turn=3, active_constraint_count=0),
        DiversificationContext(turn=1, active_constraint_count=2),
    ):
        decision = diversifier.rerank(ranking, context)
        assert not decision.activated
        assert decision.ranking == ranking
